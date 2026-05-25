"""
Publication-quality visualization script for AP-CUT research paper.

Generates all figures needed for the paper:
  Fig 1 — Training loss curves (G components + D)
  Fig 2 — Image comparison grid (clean / degraded / AP-CUT / CycleGAN)
  Fig 3 — Metrics bar chart (Accuracy, F1, SSIM, PSNR before/after)
  Fig 4 — Confusion matrices (degraded vs adapted)
  Fig 5 — ROC curves (per class, all models)
  Fig 6 — Anatomy preservation (image + segmentation mask overlay)

All figures are saved as high-resolution PNG (300 DPI) in figures/

Usage:
    python scripts/visualize_results.py \
        --cut_ckpt   checkpoints/cut/epoch_0200.pth \
        --diag_ckpt  checkpoints/diagnostic_best.pth \
        --out_dir    figures
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    accuracy_score, f1_score, ConfusionMatrixDisplay
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.dataset import DiagnosticDataset
from src.data.degradation import DegradationPipeline
from src.data.transforms import get_transforms
from src.models.diagnostic_model import DiagnosticModel

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Normal", "Benign", "Malignant"]
COLORS      = ["#2196F3", "#4CAF50", "#F44336"]   # blue, green, red
DPI         = 300

# ── Matplotlib global style ─────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       11,
    "axes.titlesize":  13,
    "axes.labelsize":  12,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":      DPI,
    "savefig.dpi":     DPI,
    "savefig.bbox":    "tight",
    "savefig.pad_inches": 0.1,
})


# ══════════════════════════════════════════════════════════════════════════════
#  Fig 1 — Training Loss Curves
# ══════════════════════════════════════════════════════════════════════════════

def plot_loss_curves(csv_path: str, out_dir: Path):
    # Find the CSV with the most training data (timestamped files take priority)
    logs_dir = Path("logs")
    best_csv = None
    best_rows = 0
    for candidate in sorted(logs_dir.glob("train_losses*.csv"), reverse=True):
        try:
            n = sum(1 for _ in open(candidate)) - 1  # subtract header
            if n > best_rows:
                best_rows = n
                best_csv = candidate
        except Exception:
            continue

    if best_csv is None:
        print(f"[visualize] No loss CSV found — skipping Fig 1")
        return

    df = pd.read_csv(best_csv)
    n_epochs = len(df)
    print(f"[visualize] Using loss CSV: {best_csv.name} ({n_epochs} epochs)")

    loss_cols = [c for c in df.columns if c != "epoch"]
    g_cols = [c for c in loss_cols if "loss_G" in c or "nce" in c or "idt" in c or "anat" in c]
    d_cols = [c for c in loss_cols if "loss_D" in c]
    other  = [c for c in loss_cols if c not in g_cols + d_cols]
    groups = [("Generator Losses", g_cols, "#1565C0"),
              ("Discriminator Loss", d_cols, "#B71C1C"),
              ("Other", other, "#2E7D32")]

    n_panels = sum(1 for _, cols, _ in groups if cols)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    ax_idx = 0
    for title, cols, base_color in groups:
        if not cols:
            continue
        ax = axes[ax_idx]
        cmap = plt.cm.Blues if "Generator" in title else (
               plt.cm.Reds if "Discriminator" in title else plt.cm.Greens)
        for i, col in enumerate(cols):
            shade = 0.4 + 0.5 * (i / max(len(cols) - 1, 1))
            ax.plot(df["epoch"], df[col], label=col.replace("loss_", ""),
                    color=cmap(shade), linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=9, framealpha=0.5)
        if n_epochs < 20:
            ax.text(0.5, 0.97, f"⚠ Only {n_epochs} epochs in log",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=8, color="red",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff3cd", alpha=0.8))
        ax_idx += 1

    fig.suptitle("AP-CUT Training Loss Curves", fontsize=14, fontweight="bold", y=1.02)
    _save(fig, out_dir / "fig1_loss_curves.png")
    print("[visualize] Fig 1 saved -> fig1_loss_curves.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Fig 2 — Image Comparison Grid
# ══════════════════════════════════════════════════════════════════════════════

def plot_image_grid(adapter_cut, adapter_cyc, out_dir: Path, n_samples: int = 5):
    transform   = get_transforms(256, "val")
    degrade     = DegradationPipeline()
    processed   = Path("data/processed")
    all_images  = []
    for cls in ["benign", "malignant", "normal"]:
        imgs = list((processed / cls).glob("*.png"))
        all_images += imgs

    if not all_images:
        print("[visualize] No processed images found — skipping Fig 2")
        return

    # Pick n_samples evenly spaced (one per class if possible)
    idxs    = np.linspace(0, len(all_images) - 1, n_samples, dtype=int)
    samples = [all_images[i] for i in idxs]

    rows = 4 if adapter_cyc else 3
    row_labels = ["(a) Clean (HQ)", "(b) Degraded (LQ)", "(c) AP-CUT (Ours)"]
    if adapter_cyc:
        row_labels.append("(d) CycleGAN")

    fig, axes = plt.subplots(rows, n_samples,
                             figsize=(3 * n_samples, 3.2 * rows + 0.4))

    for col, path in enumerate(samples):
        clean    = np.array(Image.open(path).convert("RGB"))
        degraded = degrade(clean)

        t_deg = transform(image=degraded)["image"].unsqueeze(0).to(DEVICE)
        t_cln = transform(image=clean)["image"].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            adapted_cut = adapter_cut.translate(t_deg) if adapter_cut else t_deg
            adapted_cyc = adapter_cyc.translate(t_deg) if adapter_cyc else None

        row_imgs = [clean, degraded, _tensor_to_uint8(adapted_cut)]
        if adapter_cyc:
            row_imgs.append(_tensor_to_uint8(adapted_cyc))

        for row, img in enumerate(row_imgs):
            ax = axes[row, col]
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            ax.axis("off")

    # Row labels using fig.text (reliable across all layouts)
    row_colors = ["#1565C0", "#B71C1C", "#2E7D32", "#6A1B9A"]
    for row, (label, color) in enumerate(zip(row_labels, row_colors)):
        # Calculate y position for each row centre
        y_pos = 1.0 - (row + 0.55) / rows
        fig.text(0.01, y_pos, label,
                 ha="left", va="center", fontsize=10, fontweight="bold",
                 color=color, rotation=90,
                 transform=fig.transFigure)

    fig.suptitle("Domain Adaptation — Visual Comparison", fontsize=14,
                 fontweight="bold")
    plt.subplots_adjust(left=0.06, hspace=0.05, wspace=0.05)
    _save(fig, out_dir / "fig2_image_comparison.png")
    print("[visualize] Fig 2 saved -> fig2_image_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Fig 3 — Metrics Bar Chart
# ══════════════════════════════════════════════════════════════════════════════

def plot_metrics_bar(metrics_dict: dict, out_dir: Path):
    """
    Only classification + SSIM metrics are shown in the bar chart.
    PSNR is excluded — it is misleading in style-transfer tasks where
    pixel-level deviation from the clean image is expected.
    """
    if not metrics_dict:
        print("[visualize] No metrics provided — skipping Fig 3")
        return

    # Keep only Accuracy, Macro-F1, SSIM — drop PSNR (norm) from bar chart
    SHOW_METRICS = ["Accuracy", "Macro-F1", "SSIM"]
    models     = list(metrics_dict.keys())
    bar_colors = ["#90CAF9", "#A5D6A7", "#EF9A9A", "#CE93D8"]

    # Filter to available metrics
    available = [m for m in SHOW_METRICS
                 if m in next(iter(metrics_dict.values()))]
    x     = np.arange(len(available))
    width = 0.7 / len(models)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (model, vals) in enumerate(metrics_dict.items()):
        offsets = x + (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(offsets, [vals.get(m, 0) for m in available], width,
                      label=model, color=bar_colors[i % len(bar_colors)],
                      edgecolor="white", linewidth=0.8)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(available, fontsize=12)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Quantitative Comparison Across Models", fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    # Reference line at 0.8
    ax.axhline(y=0.8, color="gray", linestyle="--", linewidth=1.0, alpha=0.6,
               label="_0.8 target")
    ax.text(len(available) - 0.45, 0.81, "0.80 target",
            fontsize=8, color="gray", va="bottom")

    _save(fig, out_dir / "fig3_metrics_bar.png")
    print("[visualize] Fig 3 saved -> fig3_metrics_bar.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Fig 4 — Confusion Matrices
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrices(results: dict, out_dir: Path):
    """
    results = {"Degraded": (y_true, y_pred), "AP-CUT": (y_true, y_pred), ...}
    """
    if not results:
        print("[visualize] No predictions — skipping Fig 4")
        return

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    from matplotlib.patches import FancyBboxPatch
    import matplotlib.cm as mplcm
    import matplotlib.colors as mcolors

    for i, (ax, (name, (y_true, y_pred))) in enumerate(zip(axes, results.items())):
        cm      = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

        # Draw manually with imshow — gives full control over labels
        cmap = plt.cm.Blues
        ax.imshow(cm_norm, interpolation="nearest", cmap=cmap,
                  vmin=0, vmax=1)

        # Cell text: percentage + raw count
        thresh = 0.5
        for row in range(cm.shape[0]):
            for col_idx in range(cm.shape[1]):
                val  = cm_norm[row, col_idx]
                cnt  = cm[row, col_idx]
                color = "white" if val > thresh else "#1a237e"
                ax.text(col_idx, row - 0.1, f"{val:.0%}",
                        ha="center", va="center",
                        fontsize=11, fontweight="bold", color=color)
                ax.text(col_idx, row + 0.25, f"n={cnt}",
                        ha="center", va="center",
                        fontsize=8, color=color, alpha=0.85)

        # Axes formatting
        ticks = np.arange(len(CLASS_NAMES))
        ax.set_xticks(ticks)
        ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=10)
        ax.set_yticks(ticks)
        ax.set_yticklabels(CLASS_NAMES, fontsize=10)

        # Consistent ylabel on LEFT for all subplots
        ax.set_ylabel("True label", fontsize=11, labelpad=8)
        ax.yaxis.set_label_position("left")
        ax.set_xlabel("Predicted label", fontsize=11, labelpad=8)

        acc = accuracy_score(y_true, y_pred)
        ax.set_title(f"{name}\nAcc = {acc:.3f}", fontweight="bold", fontsize=11)

        # Red border on Malignant diagonal cell
        mal_idx = CLASS_NAMES.index("Malignant")
        ax.add_patch(FancyBboxPatch(
            (mal_idx - 0.5, mal_idx - 0.5), 1, 1,
            boxstyle="round,pad=0.05",
            linewidth=2.5, edgecolor="#D32F2F",
            facecolor="none", zorder=5))

    fig.suptitle("Confusion Matrices — Before vs After Adaptation",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, out_dir / "fig4_confusion_matrices.png")
    print("[visualize] Fig 4 saved -> fig4_confusion_matrices.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Fig 5 — ROC Curves
# ══════════════════════════════════════════════════════════════════════════════

def plot_roc_curves(roc_data: dict, out_dir: Path):
    """
    roc_data = {
        "Degraded":  (y_true_onehot, y_prob),
        "AP-CUT":    (y_true_onehot, y_prob),
    }
    """
    if not roc_data:
        print("[visualize] No ROC data — skipping Fig 5")
        return

    fig, axes = plt.subplots(1, len(CLASS_NAMES),
                             figsize=(5 * len(CLASS_NAMES), 4.5))
    line_styles = ["-", "--", "-."]
    model_colors = ["#1E88E5", "#43A047", "#E53935", "#8E24AA"]

    for cls_idx, (cls_name, ax) in enumerate(zip(CLASS_NAMES, axes)):
        for m_idx, (model_name, (y_oh, y_prob)) in enumerate(roc_data.items()):
            fpr, tpr, _ = roc_curve(y_oh[:, cls_idx], y_prob[:, cls_idx])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr,
                    label=f"{model_name} (AUC={roc_auc:.3f})",
                    color=model_colors[m_idx % len(model_colors)],
                    linestyle=line_styles[m_idx % len(line_styles)],
                    linewidth=2)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4)
        ax.set_xlabel("False Positive Rate", fontsize=10)
        ax.set_ylabel("True Positive Rate", fontsize=10)
        ax.set_title(f"ROC — {cls_name}", fontweight="bold")
        ax.legend(fontsize=9, loc="lower right",
                  framealpha=0.9, edgecolor="lightgray")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.fill_between([0, 1], [0, 1], alpha=0.04, color="gray")

    fig.suptitle("ROC Curves per Class", fontsize=13, fontweight="bold", y=1.02)
    _save(fig, out_dir / "fig5_roc_curves.png")
    print("[visualize] Fig 5 saved -> fig5_roc_curves.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Fig 6 — Anatomy Preservation
# ══════════════════════════════════════════════════════════════════════════════

def plot_anatomy_preservation(adapter_cut, out_dir: Path, n_samples: int = 4):
    from src.losses.anatomy import AnatomyPreservingLoss

    transform  = get_transforms(256, "val")
    degrade    = DegradationPipeline()
    processed  = Path("data/processed")
    all_images = list(processed.rglob("*.png"))
    if not all_images:
        print("[visualize] No images for anatomy plot — skipping Fig 6")
        return

    # Pick samples that have actual lesions (benign/malignant only, not normal)
    lesion_images = [p for p in all_images
                     if "normal" not in str(p).lower()]
    if len(lesion_images) < n_samples:
        lesion_images = all_images   # fallback
    idxs    = np.linspace(0, len(lesion_images) - 1, n_samples, dtype=int)
    samples = [lesion_images[i] for i in idxs]

    seg_loss = AnatomyPreservingLoss(
        segmentor_type="unet",
        checkpoint="checkpoints/seg_unet.pth",
        device=DEVICE,
    )

    fig, axes = plt.subplots(3, n_samples,
                             figsize=(3.2 * n_samples, 9.5))
    row_labels  = ["(a) Degraded Input", "(b) AP-CUT Output",
                   "(c) Lesion Mask Overlay"]
    row_colors  = ["#1565C0", "#2E7D32", "#B71C1C"]

    MASK_THRESH = 0.35   # only highlight high-confidence lesion regions

    for col, path in enumerate(samples):
        clean    = np.array(Image.open(path).convert("RGB"))
        degraded = degrade(clean)
        t_deg    = transform(image=degraded)["image"].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            adapted  = adapter_cut.translate(t_deg) if adapter_cut else t_deg
            mask_adp = seg_loss.seg((adapted + 1) / 2).squeeze().cpu().numpy()

        deg_np = _tensor_to_uint8(t_deg)
        adp_np = _tensor_to_uint8(adapted)

        # Threshold mask — only show where confidence > MASK_THRESH
        mask_bin = np.where(mask_adp > MASK_THRESH, mask_adp, 0.0)

        # Overlay: red channel boost only where mask is confident
        overlay = adp_np.copy().astype(float)
        alpha   = 0.55
        overlay[:, :, 0] = np.clip(
            overlay[:, :, 0] + mask_bin * 255 * alpha, 0, 255)
        overlay[:, :, 1] = np.clip(
            overlay[:, :, 1] * (1 - mask_bin * 0.7), 0, 255)
        overlay[:, :, 2] = np.clip(
            overlay[:, :, 2] * (1 - mask_bin * 0.7), 0, 255)

        for row, img in enumerate([deg_np, adp_np, overlay.astype(np.uint8)]):
            ax = axes[row, col]
            ax.imshow(img)
            ax.axis("off")

    # Row labels via fig.text (reliable)
    for row, (label, color) in enumerate(zip(row_labels, row_colors)):
        y_pos = 1.0 - (row + 0.55) / 3
        fig.text(0.01, y_pos, label,
                 ha="left", va="center", fontsize=10, fontweight="bold",
                 color=color, rotation=90,
                 transform=fig.transFigure)

    # Legend
    legend_elements = [
        Patch(facecolor="#D32F2F", alpha=0.7, label=f"Predicted Lesion (conf > {MASK_THRESH})")]
    fig.legend(handles=legend_elements, loc="lower center",
               bbox_to_anchor=(0.5, 0.01), fontsize=10, framealpha=0.9)
    fig.suptitle("Anatomy Preservation — Lesion Structure Maintained After Adaptation",
                 fontsize=13, fontweight="bold")
    plt.subplots_adjust(left=0.07, top=0.94, hspace=0.04, wspace=0.04)
    _save(fig, out_dir / "fig6_anatomy_preservation.png")
    print("[visualize] Fig 6 saved -> fig6_anatomy_preservation.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Helper utilities
# ══════════════════════════════════════════════════════════════════════════════

def _tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    img = t.squeeze(0).cpu().float()
    img = (img * 0.5 + 0.5).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _save(fig: plt.Figure, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _collect_predictions(adapter, diag_model, n_samples: int = 780):
    """Run the full pipeline and return (y_true, y_pred, y_prob) arrays."""
    import random as _random
    from torch.nn.functional import softmax

    _random.seed(42)
    np.random.seed(42)

    transform  = get_transforms(256, "val")
    degrade    = DegradationPipeline()
    label_map  = {"normal": 0, "benign": 1, "malignant": 2}
    all_paths  = []
    for cls, lbl in label_map.items():
        imgs = sorted((Path("data/processed") / cls).glob("*.png"))
        all_paths += [(p, lbl) for p in imgs]

    idxs = np.random.choice(len(all_paths), min(n_samples, len(all_paths)), replace=False)
    samples = [all_paths[i] for i in idxs]

    y_true, y_pred, y_prob = [], [], []
    for path, label in tqdm(samples, desc="  collecting preds"):
        clean    = np.array(Image.open(path).convert("RGB"))
        degraded = degrade(clean)
        t_deg    = transform(image=degraded)["image"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            adapted = adapter.translate(t_deg) if adapter else t_deg
            logits  = diag_model(adapted)
            probs   = softmax(logits, dim=-1).squeeze().cpu().numpy()
            pred    = probs.argmax()
        y_true.append(label)
        y_pred.append(pred)
        y_prob.append(probs)

    return np.array(y_true), np.array(y_pred), np.array(y_prob)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[visualize] Output directory: {out_dir.resolve()}")
    print(f"[visualize] Device: {DEVICE}")

    # ── Load models ─────────────────────────────────────────────────────────
    with open("config/train.yaml") as f:
        cfg = yaml.safe_load(f)
    with open("config/cut.yaml") as f:
        cfg.update(yaml.safe_load(f))

    diag_model = None
    adapter_cut = None
    adapter_cyc = None

    if args.diag_ckpt and Path(args.diag_ckpt).exists():
        diag_model = DiagnosticModel(checkpoint_path=args.diag_ckpt).to(DEVICE)
        print(f"[visualize] Loaded diagnostic model: {args.diag_ckpt}")

    if args.cut_ckpt and Path(args.cut_ckpt).exists():
        from src.models.cut_model import CUTModel
        adapter_cut = CUTModel(cfg, DEVICE)
        adapter_cut.load(args.cut_ckpt)
        adapter_cut.G.eval()
        print(f"[visualize] Loaded AP-CUT: {args.cut_ckpt}")

    if args.cyclegan_ckpt and Path(args.cyclegan_ckpt).exists():
        from src.models.cyclegan_model import CycleGANModel
        with open("config/cyclegan.yaml") as f:
            cyc_cfg = yaml.safe_load(f)
        cyc_cfg.update(cfg)
        adapter_cyc = CycleGANModel(cyc_cfg, DEVICE)
        ckpt = torch.load(args.cyclegan_ckpt, map_location=DEVICE)
        adapter_cyc.G_A.load_state_dict(ckpt["G_A"])
        adapter_cyc.G_A.eval()
        print(f"[visualize] Loaded CycleGAN: {args.cyclegan_ckpt}")

    # ── Fig 1 — Loss curves ──────────────────────────────────────────────────
    plot_loss_curves(args.loss_csv, out_dir)

    # ── Fig 2 — Image grid ───────────────────────────────────────────────────
    plot_image_grid(adapter_cut, adapter_cyc, out_dir)

    # ── Collect predictions for Figs 3-5 via Evaluator (same code path as run_evaluation.py) ──
    from src.evaluation.evaluate import Evaluator
    from sklearn.preprocessing import label_binarize
    metrics_dict, confusion_results, roc_data = {}, {}, {}

    if diag_model:
        for label, adapter in [("Degraded (No Adapt)", None),
                                ("AP-CUT (Ours)", adapter_cut),
                                ("CycleGAN", adapter_cyc)]:
            if adapter is None and label != "Degraded (No Adapt)":
                continue
            if adapter is None:
                # Degrade-only baseline: use a pass-through adapter
                class _PassThrough:
                    def translate(self, x): return x
                _adapter = _PassThrough()
            else:
                _adapter = adapter
            print(f"[visualize] Evaluating: {label}")
            ev = Evaluator(_adapter, diag_model, device=DEVICE)
            m  = ev.evaluate("data/processed")

            metrics_dict[label] = {
                "Accuracy": round(m["accuracy_adapted"], 4),
                "Macro-F1": round(m["f1_adapted"],       4),
                "SSIM":     round(m["mean_ssim"],        4),
                "_PSNR_dB": round(m["mean_psnr"],        2),
            }
            confusion_results[label] = (m["y_true"], m["y_pred_adp"])
            y_oh = label_binarize(m["y_true"], classes=[0, 1, 2])
            roc_data[label] = (y_oh, m["y_prob_adp"])

    # ── Fig 3-5 ──────────────────────────────────────────────────────────────
    plot_metrics_bar(metrics_dict, out_dir)
    plot_confusion_matrices(confusion_results, out_dir)
    plot_roc_curves(roc_data, out_dir)

    # ── Fig 6 — Anatomy preservation ─────────────────────────────────────────
    if adapter_cut:
        plot_anatomy_preservation(adapter_cut, out_dir)

    print(f"\n[visualize] All figures saved to: {out_dir.resolve()}")
    print("[visualize] Files:")
    for f in sorted(out_dir.glob("fig*.png")):
        size_kb = f.stat().st_size // 1024
        print(f"   {f.name}  ({size_kb} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cut_ckpt",      default=None,
                        help="Path to AP-CUT checkpoint")
    parser.add_argument("--cyclegan_ckpt", default=None,
                        help="Path to CycleGAN checkpoint (optional)")
    parser.add_argument("--diag_ckpt",     default="checkpoints/diagnostic_best.pth",
                        help="Path to diagnostic model checkpoint")
    parser.add_argument("--loss_csv",      default="logs/train_losses.csv",
                        help="Path to training loss CSV")
    parser.add_argument("--out_dir",       default="figures",
                        help="Output directory for figures")
    main(parser.parse_args())
