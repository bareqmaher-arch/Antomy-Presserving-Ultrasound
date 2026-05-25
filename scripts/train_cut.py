"""
Step 2: Train AP-CUT (Anatomy-Preserving Contrastive Unpaired Translation).

Trains the domain adaptation network to translate low-quality (degraded)
ultrasound images to the high-quality domain, with anatomy preservation.

Usage:
    python scripts/train_cut.py [--config config/cut.yaml] [--resume checkpoints/epoch_0050.pth]
"""
import sys
import argparse
import yaml
import random
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.dataset import UnpairedUltrasoundDataset
from src.models.cut_model import CUTModel
from src.training.trainer import Trainer
from src.training.callbacks import CheckpointCallback, LRSchedulerCallback, WandBCallback

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cfg(train_yaml: str, model_yaml: str) -> dict:
    with open(train_yaml) as f:
        cfg = yaml.safe_load(f)
    with open(model_yaml) as f:
        cfg.update(yaml.safe_load(f))
    return cfg


def main(args):
    cfg = load_cfg("config/train.yaml", args.config)
    seed = cfg.get("seed", 42)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    print(f"[train_cut] Device: {DEVICE}")
    print(f"[train_cut] Config: {args.config}")

    train_ds = UnpairedUltrasoundDataset(
        domain_a_dir="data/degraded",
        domain_b_dir="data/processed",
        image_size=cfg["data"]["image_size"],
        mode="train",
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
        drop_last=True,
    )
    print(f"[train_cut] Train dataset: {len(train_ds)} pairs")

    model = CUTModel(cfg, device=DEVICE)
    model.setup_optimizers(
        lr_G=cfg["optimizer"]["lr_G"],
        lr_D=cfg["optimizer"]["lr_D"],
        beta1=cfg["optimizer"]["beta1"],
    )

    callbacks = [
        CheckpointCallback(
            save_dir="checkpoints/cut",
            save_interval=cfg["training"]["save_interval"],
        ),
        LRSchedulerCallback(
            model=model,
            total_epochs=cfg["training"]["epochs"],
            decay_start_epoch=cfg["scheduler"]["decay_start_epoch"],
        ),
        WandBCallback(
            project=cfg["wandb"]["project"],
            entity=cfg["wandb"].get("entity"),
            enabled=cfg["wandb"]["enabled"],
        ),
    ]

    start_epoch = 1
    if args.resume:
        start_epoch = model.load(args.resume) + 1
        print(f"[train_cut] Resumed from {args.resume}, starting at epoch {start_epoch}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        epochs=cfg["training"]["epochs"],
        device=DEVICE,
        amp=cfg["training"]["amp"],
        grad_accum_steps=cfg["training"]["gradient_accumulation_steps"],
        log_interval=cfg["training"]["log_interval"],
        callbacks=callbacks,
    )
    trainer.train(start_epoch=start_epoch)
    print("\n[train_cut] Done. Next step: python scripts/run_evaluation.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/cut.yaml")
    parser.add_argument("--resume", default=None)
    main(parser.parse_args())
