"""
AP-CUT: Anatomy-Preserving Contrastive Unpaired Translation.

Extends the original CUT model (Park et al., ECCV 2020) with an anatomy-
preserving loss that constrains the generator to not move, shrink, or
hallucinate lesion structures during style translation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import ResNetGenerator, PatchGANDiscriminator, init_weights
from .patchnce import PatchNCELoss
from ..losses.adversarial import LSGANLoss
from ..losses.anatomy import AnatomyPreservingLoss


class CUTModel(nn.Module):
    """
    One-sided unpaired image translation (Low → High quality).

    Args:
        cfg: dict loaded from config/cut.yaml merged with config/train.yaml
    """

    def __init__(self, cfg: dict, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        gen_cfg = cfg["generator"]
        dis_cfg = cfg["discriminator"]
        w = cfg["loss_weights"]

        self.lambda_NCE  = w["lambda_NCE"]
        self.lambda_idt  = w["lambda_idt"]
        self.lambda_anat = w["lambda_anat"]

        # Networks
        self.G = ResNetGenerator(
            ngf=gen_cfg["ngf"],
            n_blocks=gen_cfg["n_blocks"],
            norm_type=gen_cfg["norm"],
            use_dropout=gen_cfg["use_dropout"],
        ).to(device)

        self.D = PatchGANDiscriminator(
            ndf=dis_cfg["ndf"],
            n_layers=dis_cfg["n_layers"],
            norm_type=dis_cfg["norm"],
        ).to(device)

        init_weights(self.G)
        init_weights(self.D)

        # Losses
        self.crit_adv  = LSGANLoss().to(device)
        self.crit_nce  = PatchNCELoss(
            num_patches=cfg["patchnce"]["num_patches"],
            temperature=cfg["patchnce"]["nce_temperature"],
        )
        self.crit_anat = AnatomyPreservingLoss(
            segmentor_type=cfg["anatomy"]["segmentor"],
            checkpoint=cfg["anatomy"].get("seg_checkpoint"),
            device=device,
        )

        # Optimisers (set up after construction)
        self.opt_G: torch.optim.Optimizer | None = None
        self.opt_D: torch.optim.Optimizer | None = None

    def setup_optimizers(self, lr_G: float = 2e-4, lr_D: float = 2e-4, beta1: float = 0.5):
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=lr_G, betas=(beta1, 0.999))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=lr_D, betas=(beta1, 0.999))
        # Separate scalers so D and G gradient scales don't interfere
        use_amp = self.cfg.get("training", {}).get("amp", True)
        self.scaler_G = torch.amp.GradScaler("cuda", enabled=use_amp)
        self.scaler_D = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _nce_layers(self) -> list[int]:
        return self.cfg["patchnce"].get("nce_layers", [0, 4, 8, 12, 16])

    def _compute_nce_loss(self, real_A: torch.Tensor, fake_B: torch.Tensor) -> torch.Tensor:
        feats_src = self.G.encode(real_A)
        feats_tgt = self.G.encode(fake_B)
        # Select only the requested layers
        layers = self._nce_layers()
        feats_src = [feats_src[i] for i in layers if i < len(feats_src)]
        feats_tgt = [feats_tgt[i] for i in layers if i < len(feats_tgt)]
        return self.crit_nce(feats_src, feats_tgt)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
    ) -> dict[str, float]:
        """
        One training iteration.  Returns a dict of scalar losses for logging.
        Callers should handle AMP scaling and gradient accumulation externally.
        """
        use_amp = getattr(self, "scaler_G", None) is not None
        amp_ctx = torch.autocast(device_type="cuda", enabled=use_amp)

        with amp_ctx:
            fake_B = self.G(real_A)

        # ---- Discriminator ----
        self.D.requires_grad_(True)
        self.opt_D.zero_grad()
        with amp_ctx:
            pred_real = self.D(real_B)
            pred_fake = self.D(fake_B.detach())
            loss_D = (self.crit_adv(pred_real, True) + self.crit_adv(pred_fake, False)) * 0.5
        if use_amp:
            self.scaler_D.scale(loss_D).backward()
            self.scaler_D.step(self.opt_D)
            self.scaler_D.update()
        else:
            loss_D.backward()
            self.opt_D.step()

        # ---- Generator ----
        self.D.requires_grad_(False)
        self.opt_G.zero_grad()
        with amp_ctx:
            pred_fake_G = self.D(fake_B)
            loss_G_adv  = self.crit_adv(pred_fake_G, True)
            loss_G_nce  = self._compute_nce_loss(real_A, fake_B) * self.lambda_NCE
            loss_G_idt  = F.l1_loss(self.G(real_B), real_B) * self.lambda_idt
            loss_G_anat = self.crit_anat(fake_B, real_A) * self.lambda_anat
            loss_G = loss_G_adv + loss_G_nce + loss_G_idt + loss_G_anat
        if use_amp:
            self.scaler_G.scale(loss_G).backward()
            self.scaler_G.step(self.opt_G)
            self.scaler_G.update()
        else:
            loss_G.backward()
            self.opt_G.step()

        return {
            "loss_D":    loss_D.item(),
            "loss_G_adv": loss_G_adv.item(),
            "loss_G_nce": loss_G_nce.item(),
            "loss_G_idt": loss_G_idt.item(),
            "loss_G_anat": loss_G_anat.item(),
            "loss_G":    loss_G.item(),
        }

    @torch.no_grad()
    def translate(self, x: torch.Tensor) -> torch.Tensor:
        """Inference: low-quality image → high-quality domain."""
        return self.G(x)

    def save(self, path: str, epoch: int):
        torch.save({
            "epoch": epoch,
            "G_state": self.G.state_dict(),
            "D_state": self.D.state_dict(),
            "opt_G_state": self.opt_G.state_dict() if self.opt_G else None,
            "opt_D_state": self.opt_D.state_dict() if self.opt_D else None,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.G.load_state_dict(ckpt["G_state"])
        self.D.load_state_dict(ckpt["D_state"])
        if self.opt_G and ckpt.get("opt_G_state"):
            self.opt_G.load_state_dict(ckpt["opt_G_state"])
        if self.opt_D and ckpt.get("opt_D_state"):
            self.opt_D.load_state_dict(ckpt["opt_D_state"])
        return ckpt.get("epoch", 0)
