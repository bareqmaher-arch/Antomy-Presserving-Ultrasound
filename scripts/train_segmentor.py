"""
Step 1b: Train the UNet Segmentor on BUSI ground-truth masks.

This is the CRITICAL step for validating the Anatomy-Preserving claim.
Without a trained segmentor, loss_G_anat is meaningless noise.

The segmentor is trained to detect lesion regions in ultrasound images.
After training it is FROZEN and used inside AnatomyPreservingLoss to
constrain the AP-CUT generator to preserve lesion location/shape.

Usage:
    python scripts/train_segmentor.py

Output:
    checkpoints/seg_unet.pth  ← loaded automatically by AnatomyPreservingLoss
"""
import sys
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.losses.anatomy import _UNetSegmentor

# ── Config ───────────────────────────────────────────────────────────────────
RAW_DIR      = Path("data/raw")
CKPT_PATH    = Path("checkpoints/seg_unet.pth")
IMAGE_SIZE   = 256
EPOCHS       = 60
BATCH_SIZE   = 8
LR           = 1e-3
SEED         = 42
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES      = ["benign", "malignant", "normal"]


# ── Dataset ──────────────────────────────────────────────────────────────────
class BUSISegDataset(Dataset):
    """
    Loads (image, mask) pairs from BUSI raw folders.
    Images with multiple masks (e.g. *_mask_1.png) are merged into one binary mask.
    Normal class has no lesion → all-zero mask.
    """

    def __init__(self, root: Path, image_size: int = 256, mode: str = "train"):
        self.size   = image_size
        self.mode   = mode
        self.pairs  = []
        self._build(root)

    def _build(self, root: Path):
        for cls in CLASSES:
            cls_dir = root / cls
            if not cls_dir.exists():
                continue
            images = sorted(p for p in cls_dir.iterdir()
                            if p.suffix == ".png" and "_mask" not in p.name)
            for img_path in images:
                # Collect all masks for this image (primary + _mask_1, _mask_2 …)
                stem  = img_path.stem
                masks = sorted(cls_dir.glob(f"{stem}_mask*.png"))
                self.pairs.append((img_path, masks, cls))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_paths, cls = self.pairs[idx]

        # Load image → RGB → resize → [0,1]
        img = Image.open(img_path).convert("RGB").resize(
            (self.size, self.size), Image.BILINEAR)
        img_t = torch.from_numpy(
            np.array(img, dtype=np.float32) / 255.0
        ).permute(2, 0, 1)                              # (3, H, W)

        # Normalise to [-1, 1] (same as generator input)
        img_t = img_t * 2.0 - 1.0

        # Load + merge masks
        if mask_paths and cls != "normal":
            combined = np.zeros((self.size, self.size), dtype=np.float32)
            for mp in mask_paths:
                # Convert to 'L' (grayscale) to ensure shape is (H, W) not (H, W, C)
                m = np.array(
                    Image.open(mp).convert("L").resize(
                        (self.size, self.size), Image.NEAREST),
                    dtype=np.float32)
                # 'L' gives 0..255; normalise to [0, 1]
                if m.max() > 1:
                    m = m / 255.0
                combined = np.clip(combined + m, 0, 1)
            mask_t = torch.from_numpy(combined).unsqueeze(0)    # (1, H, W)
        else:
            mask_t = torch.zeros(1, self.size, self.size)

        return img_t, mask_t


# ── Loss: Binary Cross Entropy + Dice ────────────────────────────────────────
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce_w = bce_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce  = F.binary_cross_entropy(pred, target)
        # Dice
        inter = (pred * target).sum(dim=(2, 3))
        dice  = 1 - (2 * inter + 1) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1)
        return self.bce_w * bce + (1 - self.bce_w) * dice.mean()


# ── Metrics ───────────────────────────────────────────────────────────────────
def dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_bin = (pred > threshold).float()
    inter = (pred_bin * target).sum()
    if pred_bin.sum() + target.sum() == 0:
        return 1.0
    return (2 * inter / (pred_bin.sum() + target.sum())).item()


# ── Train / Val loops ─────────────────────────────────────────────────────────
def train_epoch(model, loader, opt, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    for imgs, masks in tqdm(loader, desc="  train", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        opt.zero_grad()
        # Forward in fp16, loss in fp32 (BCE is unsafe in fp16)
        with torch.autocast(device_type="cuda"):
            preds = model(imgs)
        loss = criterion(preds.float(), masks.float())
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_dice = 0.0, 0.0
    for imgs, masks in tqdm(loader, desc="  val  ", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        total_loss += criterion(preds, masks).item()
        total_dice += dice_score(preds.cpu(), masks.cpu())
    return total_loss / len(loader), total_dice / len(loader)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    print(f"[seg] Device : {DEVICE}")

    full_ds = BUSISegDataset(RAW_DIR, IMAGE_SIZE)
    n       = len(full_ds)
    n_val   = max(1, int(n * 0.15))
    n_train = n - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    print(f"[seg] Train: {n_train} | Val: {n_val}")

    model     = _UNetSegmentor().to(DEVICE)
    opt       = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    criterion = BCEDiceLoss(bce_weight=0.5)
    scaler    = torch.amp.GradScaler("cuda")

    best_dice = 0.0
    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_epoch(model, train_loader, opt, criterion, scaler, DEVICE)
        scheduler.step()
        val_loss, val_dice = val_epoch(model, val_loader, criterion, DEVICE)

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_dice={val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"  >> Best checkpoint saved (dice={best_dice:.4f})")

    print(f"\n[seg] Done. Best val Dice: {best_dice:.4f}")
    print(f"[seg] Checkpoint: {CKPT_PATH}")
    print("[seg] Next step: python scripts/train_cut.py  (AP-CUT retrain with real anatomy loss)")
