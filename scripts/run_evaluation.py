"""
Step 3: End-to-end evaluation.

Runs the full degrade → adapt → diagnose pipeline on the test split and
prints a comparison table: baseline (no adaptation) vs AP-CUT vs CycleGAN.

Usage:
    python scripts/run_evaluation.py \
        --cut_ckpt checkpoints/cut/epoch_0200.pth \
        --cyclegan_ckpt checkpoints/cyclegan/epoch_0200.pth \
        --diag_ckpt checkpoints/diagnostic_best.pth
"""
import sys
import argparse
import yaml
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.cut_model import CUTModel
from src.models.cyclegan_model import CycleGANModel
from src.models.diagnostic_model import DiagnosticModel
from src.evaluation.evaluate import Evaluator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_full_cfg():
    with open("config/train.yaml") as f:
        cfg = yaml.safe_load(f)
    with open("config/cut.yaml") as f:
        cfg.update(yaml.safe_load(f))
    return cfg


def main(args):
    cfg = load_full_cfg()
    print(f"[eval] Device: {DEVICE}")

    diag_model = DiagnosticModel(checkpoint_path=args.diag_ckpt).to(DEVICE)

    results = {}

    # AP-CUT
    if args.cut_ckpt and Path(args.cut_ckpt).exists():
        cut_model = CUTModel(cfg, DEVICE)
        cut_model.load(args.cut_ckpt)
        cut_model.G.eval()
        ev = Evaluator(cut_model, diag_model, device=DEVICE)
        m = ev.evaluate("data/processed")
        results["AP-CUT"] = m
        ev.print_report(m)

    # CycleGAN baseline
    if args.cyclegan_ckpt and Path(args.cyclegan_ckpt).exists():
        with open("config/cyclegan.yaml") as f:
            cyc_cfg = yaml.safe_load(f)
        with open("config/train.yaml") as f:
            cyc_cfg.update(yaml.safe_load(f))
        with open("config/cut.yaml") as f:
            cut_extra = yaml.safe_load(f)
        cyc_cfg.setdefault("anatomy", cut_extra["anatomy"])
        cyc_cfg.setdefault("patchnce", cut_extra["patchnce"])

        cyc_model = CycleGANModel(cyc_cfg, DEVICE)
        ckpt = torch.load(args.cyclegan_ckpt, map_location=DEVICE)
        cyc_model.G_A.load_state_dict(ckpt["G_A"])
        cyc_model.G_A.eval()
        ev = Evaluator(cyc_model, diag_model, device=DEVICE)
        m = ev.evaluate("data/processed")
        results["CycleGAN"] = m
        ev.print_report(m)

    # Comparison table
    if len(results) > 1:
        print("\n" + "=" * 70)
        print(f"  {'Model':<15} {'Acc↑':>8} {'F1↑':>8} {'SSIM↑':>8} {'PSNR↑':>8}")
        print("-" * 70)
        for name, m in results.items():
            print(f"  {name:<15} {m['accuracy_adapted']:>8.4f} {m['f1_adapted']:>8.4f} "
                  f"{m['mean_ssim']:>8.4f} {m['mean_psnr']:>8.2f}")
        print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cut_ckpt",      default="checkpoints/cut/epoch_0200.pth")
    parser.add_argument("--cyclegan_ckpt", default="checkpoints/cyclegan/epoch_0200.pth")
    parser.add_argument("--diag_ckpt",     default="checkpoints/diagnostic_best.pth")
    main(parser.parse_args())
