"""
Step 2 (baseline): Train CycleGAN for ablation comparison against AP-CUT.

Usage:
    python scripts/train_cyclegan.py [--config config/cyclegan.yaml]
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
from src.models.cyclegan_model import CycleGANModel
from src.training.trainer import Trainer
from src.training.callbacks import CheckpointCallback, LRSchedulerCallback, WandBCallback

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cfg(train_yaml, model_yaml):
    with open(train_yaml) as f:
        cfg = yaml.safe_load(f)
    with open(model_yaml) as f:
        cfg.update(yaml.safe_load(f))
    return cfg


def main(args):
    cfg = load_cfg("config/train.yaml", args.config)
    # CycleGAN needs anatomy config too — merge from cut.yaml for anatomy block
    with open("config/cut.yaml") as f:
        cut_cfg = yaml.safe_load(f)
    cfg.setdefault("anatomy", cut_cfg["anatomy"])
    cfg.setdefault("patchnce", cut_cfg["patchnce"])

    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    print(f"[train_cyclegan] Device: {DEVICE}")

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
        pin_memory=True,
        drop_last=True,
    )

    model = CycleGANModel(cfg, device=DEVICE)
    model.setup_optimizers(
        lr_G=cfg["optimizer"]["lr_G"],
        lr_D=cfg["optimizer"]["lr_D"],
        beta1=cfg["optimizer"]["beta1"],
    )

    callbacks = [
        CheckpointCallback("checkpoints/cyclegan", cfg["training"]["save_interval"]),
        LRSchedulerCallback(model, cfg["training"]["epochs"], cfg["scheduler"]["decay_start_epoch"]),
        WandBCallback(cfg["wandb"]["project"], enabled=cfg["wandb"]["enabled"]),
    ]

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
    trainer.train()
    print("\n[train_cyclegan] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/cyclegan.yaml")
    main(parser.parse_args())
