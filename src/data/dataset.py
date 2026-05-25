"""Dataset classes for unpaired domain adaptation and supervised classification."""
import random
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import get_transforms
from .degradation import DegradationPipeline


class UnpairedUltrasoundDataset(Dataset):
    """
    Loads two image pools — domain A (degraded / low-quality) and domain B
    (clean / high-quality) — and returns random unpaired samples for CUT /
    CycleGAN training.
    """

    def __init__(
        self,
        domain_a_dir: str,
        domain_b_dir: str,
        image_size: int = 256,
        mode: str = "train",
    ):
        self.transform = get_transforms(image_size, mode)
        self.files_a = sorted(Path(domain_a_dir).rglob("*.png")) + \
                       sorted(Path(domain_a_dir).rglob("*.jpg"))
        self.files_b = sorted(Path(domain_b_dir).rglob("*.png")) + \
                       sorted(Path(domain_b_dir).rglob("*.jpg"))

        if not self.files_a:
            raise FileNotFoundError(f"No images found in domain A: {domain_a_dir}")
        if not self.files_b:
            raise FileNotFoundError(f"No images found in domain B: {domain_b_dir}")

    def __len__(self):
        return max(len(self.files_a), len(self.files_b))

    def _load(self, path: Path) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        return np.array(img)

    def __getitem__(self, idx):
        img_a = self._load(self.files_a[idx % len(self.files_a)])
        img_b = self._load(self.files_b[random.randint(0, len(self.files_b) - 1)])

        aug_a = self.transform(image=img_a)["image"]
        aug_b = self.transform(image=img_b)["image"]
        return {"A": aug_a, "B": aug_b}


class DiagnosticDataset(Dataset):
    """
    Supervised dataset for training / evaluating the diagnostic classifier.
    Classes are inferred from subdirectory names (benign / malignant / normal).
    """

    CLASS_MAP = {"normal": 0, "benign": 1, "malignant": 2}

    def __init__(self, root_dir: str, image_size: int = 256, mode: str = "train"):
        self.transform = get_transforms(image_size, mode)
        self.samples = []
        root = Path(root_dir)
        for class_name, label in self.CLASS_MAP.items():
            class_dir = root / class_name
            if class_dir.exists():
                for f in class_dir.rglob("*.png"):
                    self.samples.append((f, label))
                for f in class_dir.rglob("*.jpg"):
                    self.samples.append((f, label))

        if not self.samples:
            raise FileNotFoundError(f"No labelled images found under {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = np.array(Image.open(path).convert("RGB"))
        aug = self.transform(image=img)["image"]
        return aug, torch.tensor(label, dtype=torch.long)
