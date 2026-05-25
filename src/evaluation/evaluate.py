"""End-to-end evaluation: degrade → adapt → diagnose → report metrics."""
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ..data.degradation import DegradationPipeline
from ..data.transforms import get_transforms
from ..evaluation.metrics import compute_ssim, compute_psnr, compute_dice
from ..models.diagnostic_model import DiagnosticModel


LABEL_NAMES = {0: "normal", 1: "benign", 2: "malignant"}


class Evaluator:
    """
    Runs the full three-stage pipeline on a test split:
      1. Degrade clean images with DegradationPipeline
      2. Pass degraded images through the adaptation model (G)
      3. Classify adapted images with the frozen diagnostic model
      4. Compare to baseline (degraded without adaptation)

    Reports:
      - Accuracy / macro-F1 before and after adaptation
      - Per-image SSIM and PSNR (adapted vs clean)
      - Dice score for anatomy preservation (requires segmentor)
    """

    def __init__(
        self,
        adapter_model,          # CUTModel or CycleGANModel
        diagnostic_model: DiagnosticModel,
        image_size: int = 256,
        device: torch.device = torch.device("cuda"),
        anatomy_loss=None,      # optional AnatomyPreservingLoss for Dice computation
    ):
        self.adapter = adapter_model
        self.diagnostic = diagnostic_model
        self.device = device
        self.transform = get_transforms(image_size, "val")
        self.degrade = DegradationPipeline()
        self.anatomy_loss = anatomy_loss

    def _to_tensor(self, img_np: np.ndarray) -> torch.Tensor:
        aug = self.transform(image=img_np)["image"]
        return aug.unsqueeze(0).to(self.device)

    def _to_numpy_uint8(self, tensor: torch.Tensor) -> np.ndarray:
        img = tensor.squeeze(0).cpu().float()
        img = (img * 0.5 + 0.5).clamp(0, 1)  # [-1,1] → [0,1]
        return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    @torch.no_grad()
    def evaluate(self, test_image_dir: str, label_map: dict[str, int] | None = None) -> dict:
        """
        Args:
            test_image_dir: directory with subdirs per class (normal/benign/malignant)
            label_map: override default class→label mapping

        Returns:
            dict with keys: accuracy_degraded, accuracy_adapted, f1_degraded,
                            f1_adapted, mean_ssim, mean_psnr, mean_dice
        """
        import random as _random
        _random.seed(42)
        np.random.seed(42)

        from sklearn.metrics import accuracy_score, f1_score

        if label_map is None:
            label_map = {"normal": 0, "benign": 1, "malignant": 2}

        results = {"gt": [], "pred_deg": [], "pred_adp": [], "prob_deg": [], "prob_adp": [], "ssim": [], "psnr": [], "dice": []}

        root = Path(test_image_dir)
        image_paths = []
        for cls_name, lbl in label_map.items():
            for f in sorted((root / cls_name).rglob("*.png")):
                image_paths.append((f, lbl))
            for f in sorted((root / cls_name).rglob("*.jpg")):
                image_paths.append((f, lbl))

        for path, label in tqdm(image_paths, desc="Evaluating"):
            clean = np.array(Image.open(path).convert("RGB"))
            degraded = self.degrade(clean)

            t_clean   = self._to_tensor(clean)
            t_degraded = self._to_tensor(degraded)
            t_adapted  = self.adapter.translate(t_degraded)

            pred_deg, prob_deg = self.diagnostic.predict(t_degraded)
            pred_adp, prob_adp = self.diagnostic.predict(t_adapted)

            results["gt"].append(label)
            results["pred_deg"].append(pred_deg.item())
            results["pred_adp"].append(pred_adp.item())
            results["prob_deg"].append(prob_deg.squeeze().cpu().numpy())
            results["prob_adp"].append(prob_adp.squeeze().cpu().numpy())

            adapted_np = self._to_numpy_uint8(t_adapted)
            results["ssim"].append(compute_ssim(adapted_np, clean))
            results["psnr"].append(compute_psnr(adapted_np, clean))

        gt, pd, pa = results["gt"], results["pred_deg"], results["pred_adp"]
        return {
            "accuracy_degraded": accuracy_score(gt, pd),
            "accuracy_adapted":  accuracy_score(gt, pa),
            "f1_degraded": f1_score(gt, pd, average="macro", zero_division=0),
            "f1_adapted":  f1_score(gt, pa, average="macro", zero_division=0),
            "mean_ssim": float(np.mean(results["ssim"])),
            "mean_psnr": float(np.mean(results["psnr"])),
            "n_images":  len(image_paths),
            # raw arrays for confusion matrix and ROC curves
            "y_true":     np.array(gt),
            "y_pred_deg": np.array(pd),
            "y_pred_adp": np.array(pa),
            "y_prob_deg": np.array(results["prob_deg"]),
            "y_prob_adp": np.array(results["prob_adp"]),
        }

    def print_report(self, metrics: dict):
        print("\n" + "=" * 60)
        print("  EVALUATION REPORT — AP-CUT")
        print("=" * 60)
        print(f"  Images evaluated   : {metrics['n_images']}")
        print(f"  Accuracy (degraded): {metrics['accuracy_degraded']:.4f}")
        delta_acc = metrics['accuracy_adapted'] - metrics['accuracy_degraded']
        delta_f1  = metrics['f1_adapted'] - metrics['f1_degraded']
        print(f"  Accuracy (adapted) : {metrics['accuracy_adapted']:.4f}  ({delta_acc:+.4f})")
        print(f"  Macro-F1 (degraded): {metrics['f1_degraded']:.4f}")
        print(f"  Macro-F1 (adapted) : {metrics['f1_adapted']:.4f}  ({delta_f1:+.4f})")
        print(f"  Mean SSIM (adapted vs clean): {metrics['mean_ssim']:.4f}")
        print(f"  Mean PSNR (adapted vs clean): {metrics['mean_psnr']:.2f} dB")
        print("=" * 60 + "\n")
