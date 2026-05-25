"""
Step 0: Download BUSI dataset from Kaggle, preprocess, and generate the
simulated degraded domain.

Usage:
    python scripts/prepare_data.py

Prerequisites:
    - Kaggle API credentials in ~/.kaggle/kaggle.json
      OR manually download from:
      https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
      and unzip into data/raw/
"""
import os
import sys
import shutil
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.degradation import DegradationPipeline

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
DEGRADED_DIR  = Path("data/degraded")
IMAGE_SIZE    = 256
SEED          = 42
CLASSES       = ["benign", "malignant", "normal"]


def download_busi():
    try:
        import kaggle
        print("[prepare_data] Downloading BUSI via Kaggle API...")
        kaggle.api.dataset_download_files(
            "aryashah2k/breast-ultrasound-images-dataset",
            path=str(RAW_DIR),
            unzip=True,
        )
        print("[prepare_data] Download complete.")
    except Exception as e:
        print(f"[prepare_data] Kaggle download failed: {e}")
        print("[prepare_data] Please download manually and place in data/raw/")
        print("               URL: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset")
        sys.exit(1)


def preprocess_images():
    """Resize all raw images to IMAGE_SIZE×IMAGE_SIZE, strip masks, save as PNG."""
    print("[prepare_data] Preprocessing raw images...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for cls in CLASSES:
        (PROCESSED_DIR / cls).mkdir(parents=True, exist_ok=True)
        src_dir = RAW_DIR / cls
        if not src_dir.exists():
            # Some Kaggle zips have a nested folder
            candidates = list(RAW_DIR.rglob(cls))
            if candidates:
                src_dir = candidates[0]
            else:
                print(f"[prepare_data] WARNING: class folder '{cls}' not found in {RAW_DIR}")
                continue

        images = [f for f in src_dir.iterdir()
                  if f.suffix.lower() in (".png", ".jpg")
                  and "mask" not in f.name.lower()]

        for img_path in tqdm(images, desc=f"  {cls}"):
            img = Image.open(img_path).convert("RGB").resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS
            )
            dst = PROCESSED_DIR / cls / (img_path.stem + ".png")
            img.save(dst)

    total = sum(1 for _ in PROCESSED_DIR.rglob("*.png"))
    print(f"[prepare_data] Processed {total} images → {PROCESSED_DIR}")


def generate_degraded():
    """Apply DegradationPipeline to all processed images → DEGRADED_DIR."""
    print("[prepare_data] Generating degraded domain...")
    pipeline = DegradationPipeline()
    DEGRADED_DIR.mkdir(parents=True, exist_ok=True)

    for cls in CLASSES:
        (DEGRADED_DIR / cls).mkdir(parents=True, exist_ok=True)
        for src in tqdm(list((PROCESSED_DIR / cls).glob("*.png")), desc=f"  {cls}"):
            img = np.array(Image.open(src))
            degraded = pipeline(img)
            Image.fromarray(degraded).save(DEGRADED_DIR / cls / src.name)

    total = sum(1 for _ in DEGRADED_DIR.rglob("*.png"))
    print(f"[prepare_data] Degraded images: {total} → {DEGRADED_DIR}")


def print_summary():
    print("\n[prepare_data] Dataset Summary")
    print("-" * 40)
    for split_dir, label in [(PROCESSED_DIR, "High-quality"), (DEGRADED_DIR, "Degraded")]:
        total = 0
        for cls in CLASSES:
            n = len(list((split_dir / cls).glob("*.png")))
            total += n
            print(f"  {label} / {cls}: {n}")
        print(f"  {label} total: {total}")
    print("-" * 40)


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)

    if not any(RAW_DIR.rglob("*.png")) and not any(RAW_DIR.rglob("*.jpg")):
        download_busi()

    preprocess_images()
    generate_degraded()
    print_summary()
    print("\n[prepare_data] Done. Next step: python scripts/train_diagnostic.py")
