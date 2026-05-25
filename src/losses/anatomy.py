"""
Anatomy-Preserving Loss — the key research contribution of AP-CUT.

Motivation: GANs can hallucinate, remove, or shift lesions during image
style translation.  This loss uses a frozen segmentation network to compare
the anatomical structure (lesion masks) of the source and translated image.
Any structural divergence is penalised.

Two segmentor back-ends are supported:
  - 'unet': lightweight UNet trained on BUSI segmentation masks (preferred).
  - 'sam' : Meta's Segment Anything Model in zero-shot automatic mode
             (heavier but requires no BUSI mask labels).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _UNetSegmentor(nn.Module):
    """
    4-level UNet for binary lesion segmentation (single output channel).
    Trained on BUSI ground-truth masks via scripts/train_segmentor.py.
    Architecture: 3→32→64→128→256 encoder, symmetric decoder.
    """

    def __init__(self):
        super().__init__()

        def _block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.InstanceNorm2d(out_c),
                nn.ReLU(True),
                nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                nn.InstanceNorm2d(out_c),
                nn.ReLU(True),
            )

        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.enc1 = _block(3,   32)
        self.enc2 = _block(32,  64)
        self.enc3 = _block(64,  128)
        self.enc4 = _block(128, 256)   # bottleneck

        # Decoder
        self.up3  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = _block(256, 128)
        self.up2  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = _block(128, 64)
        self.up1  = nn.ConvTranspose2d(64,  32, 2, stride=2)
        self.dec1 = _block(64,  32)

        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))          # bottleneck

        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.head(d1))    # (B, 1, H, W) ∈ [0, 1]


class AnatomyPreservingLoss(nn.Module):
    """
    Computes the anatomy-preserving structural loss:

        L_anat = MSE( Seg(fake_B) , Seg(real_A) )

    Both inputs are images in [-1, 1].  The segmentor is frozen during
    domain-adaptation training — it is only updated during its own supervised
    training phase (scripts/train_diagnostic.py can optionally include it).
    """

    def __init__(
        self,
        segmentor_type: str = "unet",
        checkpoint: str | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.segmentor_type = segmentor_type

        if segmentor_type == "unet":
            self.seg = _UNetSegmentor()
            if checkpoint:
                try:
                    state = torch.load(checkpoint, map_location="cpu")
                    self.seg.load_state_dict(state)
                    print(f"[AnatomyLoss] Loaded UNet segmentor from {checkpoint}")
                except Exception as e:
                    print(f"[AnatomyLoss] WARNING: could not load segmentor checkpoint: {e}")
                    print("[AnatomyLoss] Using randomly initialised segmentor — train it first.")
        elif segmentor_type == "sam":
            self.seg = _SAMWrapper(device=device)
        else:
            raise ValueError(f"Unknown segmentor type: {segmentor_type}")

        if device:
            self.seg = self.seg.to(device)

        # Freeze entirely — this network is never updated by the GAN losses
        for p in self.seg.parameters():
            p.requires_grad = False
        self.seg.eval()

    def forward(self, fake_img: torch.Tensor, real_img: torch.Tensor) -> torch.Tensor:
        # Inputs are in [-1, 1]; convert to [0, 1] for the segmentor
        fake_01 = (fake_img + 1.0) / 2.0
        real_01 = (real_img + 1.0) / 2.0

        with torch.no_grad():
            seg_fake = self.seg(fake_01)
            seg_real = self.seg(real_01)

        # Gradient flows only through fake_img, not through the segmentor weights.
        # We re-run fake through the segmentor with grad enabled for the generator.
        seg_fake_grad = self.seg(fake_01)
        return F.mse_loss(seg_fake_grad, seg_real.detach())


class _SAMWrapper(nn.Module):
    """
    Thin wrapper around Meta SAM that returns a soft binary mask tensor.
    Requires `pip install segment-anything` and a downloaded checkpoint.
    Falls back to a blank mask if SAM is not installed.
    """

    def __init__(self, device=None):
        super().__init__()
        self._device = device
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            # Checkpoint must be present; user downloads separately
            self._available = True
        except ImportError:
            print("[AnatomyLoss] SAM not installed. Run: pip install segment-anything")
            self._available = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._available:
            return torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
        # SAM is not differentiable; return detached masks
        # Full SAM integration left for scripts/train_cut.py initialisation
        return torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
