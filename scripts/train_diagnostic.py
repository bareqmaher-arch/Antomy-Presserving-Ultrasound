"""
Step 1: Fine-tune ResNet50 on high-quality BUSI images.

This trains the frozen downstream diagnostic model that AP-CUT improves
accuracy on.  Run this BEFORE train_cut.py.

Usage:
    python scripts/train_diagnostic.py
"""
import sys
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))
import timm
from src.data.dataset import DiagnosticDataset

PROCESSED_DIR = "data/processed"
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

EPOCHS     = 50
LR         = 1e-4
BATCH_SIZE = 16
IMAGE_SIZE = 256
SEED       = 42
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 3


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_splits(dataset, train=0.70, val=0.15):
    n = len(dataset)
    n_train = int(n * train)
    n_val   = int(n * val)
    n_test  = n - n_train - n_val
    return random_split(dataset, [n_train, n_val, n_test],
                        generator=torch.Generator().manual_seed(SEED))


def train_epoch(model, loader, optimizer, scaler, criterion, device):
    model.train()
    total_loss = 0.0
    for imgs, labels in tqdm(loader, desc="  train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda"):
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0
    for imgs, labels in tqdm(loader, desc="  val  ", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        total_loss += criterion(logits, labels).item()
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader), acc, f1


if __name__ == "__main__":
    seed_everything(SEED)
    print(f"[diagnostic] Device: {DEVICE}")

    full_dataset = DiagnosticDataset(PROCESSED_DIR, image_size=IMAGE_SIZE, mode="train")
    train_ds, val_ds, test_ds = build_splits(full_dataset)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = timm.create_model("resnet50", pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    scaler    = torch.amp.GradScaler("cuda")

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, scaler, criterion, DEVICE)
        scheduler.step()
        val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, criterion, DEVICE)
        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_acc={val_acc:.4f} | val_f1={val_f1:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt = {"model_state": model.state_dict(), "epoch": epoch, "val_acc": val_acc}
            torch.save(ckpt, CHECKPOINT_DIR / "diagnostic_best.pth")
            print(f"  → Best checkpoint saved (acc={best_acc:.4f})")

    print(f"\n[diagnostic] Training complete. Best val acc: {best_acc:.4f}")
    print("[diagnostic] Next step: python scripts/train_cut.py")
