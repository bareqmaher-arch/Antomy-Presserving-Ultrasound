"""Image quality and anatomy preservation metrics."""
import numpy as np
import torch
from skimage.metrics import structural_similarity as _ssim
from skimage.metrics import peak_signal_noise_ratio as _psnr


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """SSIM between two uint8 HWC images."""
    return _ssim(img1, img2, channel_axis=2, data_range=255)


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    return _psnr(img1, img2, data_range=255)


def compute_dice(mask_pred: np.ndarray, mask_gt: np.ndarray, threshold: float = 0.5) -> float:
    """Dice coefficient between two soft binary mask arrays."""
    pred = (mask_pred > threshold).astype(np.float32)
    gt   = (mask_gt   > threshold).astype(np.float32)
    intersection = (pred * gt).sum()
    if pred.sum() + gt.sum() == 0:
        return 1.0  # both empty → perfect
    return 2.0 * intersection / (pred.sum() + gt.sum())


def compute_fid(real_dir: str, fake_dir: str, device: str = "cuda") -> float:
    """
    Computes FID using the pytorch-fid library.
    Lower is better (0 = identical distributions).
    """
    try:
        from pytorch_fid import fid_score
        return fid_score.calculate_fid_given_paths(
            [real_dir, fake_dir],
            batch_size=32,
            device=device,
            dims=2048,
        )
    except ImportError:
        print("[metrics] pytorch-fid not installed. Run: pip install pytorch-fid")
        return float("nan")
