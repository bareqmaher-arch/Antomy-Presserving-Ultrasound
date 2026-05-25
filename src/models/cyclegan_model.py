"""CycleGAN baseline — used for ablation against AP-CUT."""
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import ResNetGenerator, PatchGANDiscriminator, init_weights
from ..losses.adversarial import LSGANLoss
from ..losses.anatomy import AnatomyPreservingLoss


class ImagePool:
    """Stores a history of generated images to reduce discriminator oscillation."""

    def __init__(self, pool_size: int = 50):
        self.pool_size = pool_size
        self.images: list[torch.Tensor] = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        if self.pool_size == 0:
            return images
        result = []
        for img in images:
            img = img.unsqueeze(0)
            if len(self.images) < self.pool_size:
                self.images.append(img)
                result.append(img)
            elif torch.rand(1).item() > 0.5:
                idx = torch.randint(len(self.images), (1,)).item()
                result.append(self.images[idx].clone())
                self.images[idx] = img
            else:
                result.append(img)
        return torch.cat(result, dim=0)


class CycleGANModel(nn.Module):
    def __init__(self, cfg: dict, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        gen_cfg = cfg["generator"]
        dis_cfg = cfg["discriminator"]
        w = cfg["loss_weights"]
        self.lambda_cycle = w["lambda_cycle"]
        self.lambda_idt   = w["lambda_idt"]
        self.lambda_anat  = w["lambda_anat"]

        # Two generators and two discriminators
        self.G_A = ResNetGenerator(ngf=gen_cfg["ngf"], n_blocks=gen_cfg["n_blocks"]).to(device)
        self.G_B = ResNetGenerator(ngf=gen_cfg["ngf"], n_blocks=gen_cfg["n_blocks"]).to(device)
        self.D_A = PatchGANDiscriminator(ndf=dis_cfg["ndf"], n_layers=dis_cfg["n_layers"]).to(device)
        self.D_B = PatchGANDiscriminator(ndf=dis_cfg["ndf"], n_layers=dis_cfg["n_layers"]).to(device)

        for net in [self.G_A, self.G_B, self.D_A, self.D_B]:
            init_weights(net)

        self.fake_A_pool = ImagePool()
        self.fake_B_pool = ImagePool()

        self.crit_adv  = LSGANLoss().to(device)
        self.crit_anat = AnatomyPreservingLoss(
            segmentor_type=cfg["anatomy"]["segmentor"],
            checkpoint=cfg["anatomy"].get("seg_checkpoint"),
            device=device,
        )

        self.opt_G: torch.optim.Optimizer | None = None
        self.opt_D: torch.optim.Optimizer | None = None

    def setup_optimizers(self, lr_G=2e-4, lr_D=2e-4, beta1=0.5):
        self.opt_G = torch.optim.Adam(
            itertools.chain(self.G_A.parameters(), self.G_B.parameters()),
            lr=lr_G, betas=(beta1, 0.999),
        )
        self.opt_D = torch.optim.Adam(
            itertools.chain(self.D_A.parameters(), self.D_B.parameters()),
            lr=lr_D, betas=(beta1, 0.999),
        )

    def train_step(self, real_A: torch.Tensor, real_B: torch.Tensor) -> dict[str, float]:
        fake_B = self.G_A(real_A)
        rec_A  = self.G_B(fake_B)
        fake_A = self.G_B(real_B)
        rec_B  = self.G_A(fake_A)

        # Discriminators
        self.D_A.requires_grad_(True)
        self.D_B.requires_grad_(True)
        self.opt_D.zero_grad()
        loss_D_A = (self.crit_adv(self.D_A(real_A), True) +
                    self.crit_adv(self.D_A(self.fake_A_pool.query(fake_A).detach()), False)) * 0.5
        loss_D_B = (self.crit_adv(self.D_B(real_B), True) +
                    self.crit_adv(self.D_B(self.fake_B_pool.query(fake_B).detach()), False)) * 0.5
        (loss_D_A + loss_D_B).backward()
        self.opt_D.step()

        # Generators
        self.D_A.requires_grad_(False)
        self.D_B.requires_grad_(False)
        self.opt_G.zero_grad()

        loss_adv = self.crit_adv(self.D_B(fake_B), True) + self.crit_adv(self.D_A(fake_A), True)
        loss_cycle = (F.l1_loss(rec_A, real_A) + F.l1_loss(rec_B, real_B)) * self.lambda_cycle
        loss_idt   = (F.l1_loss(self.G_A(real_B), real_B) +
                      F.l1_loss(self.G_B(real_A), real_A)) * self.lambda_idt
        loss_anat  = self.crit_anat(fake_B, real_A) * self.lambda_anat

        loss_G = loss_adv + loss_cycle + loss_idt + loss_anat
        loss_G.backward()
        self.opt_G.step()

        return {
            "loss_D":     (loss_D_A + loss_D_B).item(),
            "loss_adv":   loss_adv.item(),
            "loss_cycle": loss_cycle.item(),
            "loss_idt":   loss_idt.item(),
            "loss_anat":  loss_anat.item(),
            "loss_G":     loss_G.item(),
        }

    @torch.no_grad()
    def translate(self, x: torch.Tensor) -> torch.Tensor:
        return self.G_A(x)

    def save(self, path: str, epoch: int):
        torch.save({
            "epoch": epoch,
            "G_A": self.G_A.state_dict(),
            "G_B": self.G_B.state_dict(),
            "D_A": self.D_A.state_dict(),
            "D_B": self.D_B.state_dict(),
        }, path)
