"""
Create two flowchart figures for AP-CUT paper:
  fig_flowchart1.png  — full training + inference pipeline
  fig_flowchart2.png  — AnatomyPreservingLoss detailed diagram
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import os

OUT_DIR = r"D:/PYTHON/Antomy Presserving Ultrasound/figures"

# ─────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────

def box(ax, x, y, w, h, label, sublabel=None,
        fc='#D6E4F7', ec='#2E5E9E', lw=1.5,
        fontsize=9, bold=False, radius=0.04):
    """Draw a rounded rectangle with centred label."""
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=f"round,pad={radius}",
                          facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    if sublabel:
        ax.text(x, y + h*0.12, label, ha='center', va='center',
                fontsize=fontsize, fontweight=weight, zorder=4)
        ax.text(x, y - h*0.22, sublabel, ha='center', va='center',
                fontsize=fontsize - 1.5, color='#555', zorder=4)
    else:
        ax.text(x, y, label, ha='center', va='center',
                fontsize=fontsize, fontweight=weight, zorder=4)

def arr(ax, x0, y0, x1, y1, color='#333333', lw=1.4,
        style='->', dashed=False, label=None, label_offset=(0, 0.05)):
    """Draw an arrow between two points."""
    ls = '--' if dashed else '-'
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, linestyle=ls),
                zorder=5)
    if label:
        mx, my = (x0+x1)/2 + label_offset[0], (y0+y1)/2 + label_offset[1]
        ax.text(mx, my, label, ha='center', va='bottom',
                fontsize=7.5, color=color, zorder=6)

def dashed_region(ax, x, y, w, h, label, color='#2E5E9E'):
    """Draw a labelled dashed region."""
    rect = plt.Rectangle((x, y), w, h,
                          fill=False, edgecolor=color,
                          linewidth=1.5, linestyle='--', zorder=1)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 0.05, label, ha='center', va='top',
            fontsize=9, color=color, fontweight='bold', zorder=2)


# ═══════════════════════════════════════════════════════════════
# FIGURE 1 — Full AP-CUT Pipeline
# ═══════════════════════════════════════════════════════════════

fig1, ax1 = plt.subplots(figsize=(14, 7))
ax1.set_xlim(0, 14)
ax1.set_ylim(0, 7)
ax1.axis('off')
ax1.set_aspect('equal')
ax1.set_facecolor('white')
fig1.patch.set_facecolor('white')

# ── Dashed regions ──────────────────────────────────────────
dashed_region(ax1, 0.3, 1.5, 9.4, 5.2, 'TRAINING PHASE', color='#2E5E9E')
dashed_region(ax1, 10.0, 1.5, 3.7, 5.2, 'INFERENCE PHASE', color='#2A7A2A')

# ── Training phase nodes (y positions) ──────────────────────
# Row 1 (top): Domain A, G, Fake B, D, Domain B
y_top = 5.7
bw, bh = 1.3, 0.55

box(ax1, 1.1, y_top, bw, bh, 'Domain A\n(Degraded US)',
    fc='#C8DCF5', ec='#2E5E9E', fontsize=8)
box(ax1, 2.8, y_top, bw, bh, 'Generator G',
    fc='#C8F5DA', ec='#1A7A40', fontsize=8, bold=True)
box(ax1, 4.5, y_top, bw, bh, 'Fake B\n(G(xA))',
    fc='#E8D0F5', ec='#6A1A9A', fontsize=8)
box(ax1, 6.2, y_top, bw, bh, 'Discriminator D',
    fc='#F5E6C8', ec='#9A6A00', fontsize=8, bold=True)
box(ax1, 7.9, y_top, bw, bh, 'Domain B\n(Clean US)',
    fc='#C8F5DA', ec='#1A7A40', fontsize=8)

# Row 2 (mid): Segmentor S, MSE
y_mid = 4.0
box(ax1, 2.8, y_mid, 1.5, 0.55, 'Frozen\nSegmentor S',
    fc='#FFD6E7', ec='#C00060', fontsize=8)
box(ax1, 4.5, y_mid, 1.5, 0.55, 'Frozen\nSegmentor S',
    fc='#FFD6E7', ec='#C00060', fontsize=8)

# Shared-weights label between S boxes
ax1.annotate('', xy=(4.5 - 0.75, y_mid), xytext=(2.8 + 0.75, y_mid),
             arrowprops=dict(arrowstyle='<->', color='#C00060', lw=1.2,
                             linestyle='--'), zorder=5)
ax1.text(3.65, y_mid + 0.32, 'shared weights', ha='center',
         fontsize=7, color='#C00060', style='italic')

# Row 3: Loss boxes
y_loss = 2.5
box(ax1, 2.8, y_loss, 1.3, 0.50, 'L_NCE\n(PatchNCE)',
    fc='#FFE8C0', ec='#E07000', fontsize=8)
box(ax1, 4.5, y_loss, 1.3, 0.50, 'L_anat\n(Anatomy)',
    fc='#FFE8C0', ec='#E07000', fontsize=8)
box(ax1, 6.2, y_loss, 1.3, 0.50, 'L_GAN\n(LSGAN)',
    fc='#FFE8C0', ec='#E07000', fontsize=8)
box(ax1, 7.4, y_loss, 1.5, 0.50, 'L_total',
    fc='#F5C8C8', ec='#C00000', fontsize=9, bold=True)

# Row 4 — Backprop to G
y_back = 1.7
ax1.text(4.5, y_back, '← Gradient update to Generator G only (S frozen)',
         ha='center', va='center', fontsize=7.8, color='#1A7A40',
         style='italic',
         bbox=dict(boxstyle='round,pad=0.2', fc='#E8F8EE', ec='#1A7A40', lw=1))

# ── Training arrows ─────────────────────────────────────────
arr(ax1, 1.1+0.65, y_top, 2.8-0.65, y_top, color='#333')           # A → G
arr(ax1, 2.8+0.65, y_top, 4.5-0.65, y_top, color='#333')           # G → FakeB
arr(ax1, 4.5+0.65, y_top, 6.2-0.65, y_top, color='#333')           # FakeB → D
arr(ax1, 7.9-0.65, y_top, 6.2+0.65, y_top, color='#1A7A40')        # DomB → D

# A → S (dashed, down)
arr(ax1, 1.1+0.0, y_top-0.28, 2.8-0.2, y_mid+0.28,
    color='#C00060', dashed=True)
# FakeB → S (dashed, down)
arr(ax1, 4.5+0.0, y_top-0.28, 4.5+0.0, y_mid+0.28,
    color='#C00060', dashed=True)

# S → L_anat (down)
arr(ax1, 2.8, y_mid-0.28, 2.8, y_loss+0.25, color='#E07000')
arr(ax1, 4.5, y_mid-0.28, 4.5, y_loss+0.25, color='#E07000')

# G → L_NCE (down)
arr(ax1, 2.8, y_top-0.28, 2.8, y_loss+0.25, color='#E07000', dashed=True)

# D → L_GAN (down)
arr(ax1, 6.2, y_top-0.28, 6.2, y_loss+0.25, color='#E07000', dashed=True)

# Loss → L_total
arr(ax1, 2.8+0.65, y_loss, 7.4-0.75, y_loss, color='#C00000')
arr(ax1, 4.5+0.65, y_loss, 7.4-0.75, y_loss+0.1, color='#C00000')
arr(ax1, 6.2+0.65, y_loss, 7.4-0.75, y_loss-0.05, color='#C00000')

# L_total → backprop label
arr(ax1, 7.4, y_loss-0.25, 4.5, y_back+0.15, color='#1A7A40', dashed=True)

# ── Inference phase ──────────────────────────────────────────
y_inf_top = 5.7
y_inf_mid = 4.0
y_inf_bot = 2.5
inf_bw, inf_bh = 1.4, 0.6

box(ax1, 10.9, y_inf_top, inf_bw, inf_bh, 'Degraded US\n(Handheld)',
    fc='#C8DCF5', ec='#2E5E9E', fontsize=8)
box(ax1, 10.9, y_inf_mid, inf_bw, inf_bh, 'Generator G\n(Trained)',
    fc='#C8F5DA', ec='#1A7A40', fontsize=8, bold=True)
box(ax1, 10.9, y_inf_bot, inf_bw, inf_bh, 'Adapted US\n(Clinical Quality)',
    fc='#E8D0F5', ec='#6A1A9A', fontsize=8)
box(ax1, 13.1, y_inf_mid, 0.9, 0.55, 'ResNet-50\n(Frozen)',
    fc='#F5E6C8', ec='#9A6A00', fontsize=7)
ax1.text(13.1, y_inf_bot, '→ Diagnosis\n(Normal / Benign /\nMalignant)',
         ha='center', va='center', fontsize=7.5, color='#2A7A2A',
         bbox=dict(boxstyle='round,pad=0.2', fc='#E8F8EE', ec='#2A7A2A', lw=1))

arr(ax1, 10.9, y_inf_top-0.30, 10.9, y_inf_mid+0.30, color='#333')
arr(ax1, 10.9, y_inf_mid-0.30, 10.9, y_inf_bot+0.30, color='#333')
arr(ax1, 10.9+0.70, y_inf_mid, 13.1-0.45, y_inf_mid, color='#9A6A00')
arr(ax1, 13.1, y_inf_mid-0.28, 13.1, y_inf_bot+0.3, color='#2A7A2A')

# ── Legend ───────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor='#C8DCF5', edgecolor='#2E5E9E', label='Data Domain'),
    mpatches.Patch(facecolor='#C8F5DA', edgecolor='#1A7A40', label='Generator G'),
    mpatches.Patch(facecolor='#FFD6E7', edgecolor='#C00060', label='Frozen Segmentor S'),
    mpatches.Patch(facecolor='#FFE8C0', edgecolor='#E07000', label='Loss Component'),
    mpatches.Patch(facecolor='#F5C8C8', edgecolor='#C00000', label='Total Loss'),
]
ax1.legend(handles=legend_items, loc='lower left', fontsize=7.5,
           framealpha=0.9, bbox_to_anchor=(0.01, 0.01))

ax1.set_title('AP-CUT Training and Inference Pipeline',
              fontsize=12, fontweight='bold', pad=8)

out1 = os.path.join(OUT_DIR, 'fig_flowchart1.png')
fig1.savefig(out1, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig1)
print(f"Saved: {out1}")


# ═══════════════════════════════════════════════════════════════
# FIGURE 2 — AnatomyPreservingLoss Detailed Diagram
# ═══════════════════════════════════════════════════════════════

fig2, ax2 = plt.subplots(figsize=(11, 7))
ax2.set_xlim(0, 11)
ax2.set_ylim(0, 7)
ax2.axis('off')
ax2.set_aspect('equal')
ax2.set_facecolor('white')
fig2.patch.set_facecolor('white')

bw2, bh2 = 1.6, 0.60

# ── Row 1: real_A  →  Generator G  →  fake_B ────────────────
y1 = 6.0
box(ax2, 1.5, y1, bw2, bh2, 'x_A  (real_A)\nDegraded US',
    fc='#C8DCF5', ec='#2E5E9E', fontsize=9)
box(ax2, 4.0, y1, bw2, bh2, 'Generator G',
    fc='#C8F5DA', ec='#1A7A40', fontsize=10, bold=True)
box(ax2, 6.5, y1, bw2, bh2, 'G(x_A)  (fake_B)\nAdapted US',
    fc='#E8D0F5', ec='#6A1A9A', fontsize=9)

arr(ax2, 1.5+0.80, y1, 4.0-0.80, y1, color='#333')
arr(ax2, 4.0+0.80, y1, 6.5-0.80, y1, color='#333')

# ── Row 2: Frozen UNet S (two instances sharing weights) ─────
y2 = 4.4
box(ax2, 2.0, y2, bw2+0.2, bh2+0.1, 'Frozen UNet S\n(Segmentor)',
    fc='#FFD6E7', ec='#C00060', fontsize=9, bold=True)
box(ax2, 6.5, y2, bw2+0.2, bh2+0.1, 'Frozen UNet S\n(Segmentor)',
    fc='#FFD6E7', ec='#C00060', fontsize=9, bold=True)

# Shared weights brace
ax2.annotate('', xy=(6.5-0.90, y2 + 0.42), xytext=(2.0+0.90, y2 + 0.42),
             arrowprops=dict(arrowstyle='<->', color='#C00060',
                             lw=1.3, linestyle='--'), zorder=5)
ax2.text(4.25, y2+0.60, 'shared frozen weights  (no gradient)',
         ha='center', fontsize=8, color='#C00060', style='italic')

# x_A → S (dashed)
arr(ax2, 1.5, y1-0.30, 2.0, y2+0.35,
    color='#C00060', dashed=True, label='S(x_A)', label_offset=(-0.55, 0.0))
# fake_B → S (dashed)
arr(ax2, 6.5, y1-0.30, 6.5, y2+0.35,
    color='#C00060', dashed=True, label='S(G(x_A))', label_offset=(0.65, 0.0))

# ── Row 3: segmentation mask outputs ─────────────────────────
y3 = 2.9
box(ax2, 2.0, y3, bw2+0.3, bh2, 'S(x_A)\nAnatomy Mask (source)',
    fc='#FFF5CC', ec='#B8860B', fontsize=8.5)
box(ax2, 6.5, y3, bw2+0.3, bh2, 'S(G(x_A))\nAnatomy Mask (adapted)',
    fc='#FFF5CC', ec='#B8860B', fontsize=8.5)

arr(ax2, 2.0, y2-0.35, 2.0, y3+0.30, color='#B8860B')
arr(ax2, 6.5, y2-0.35, 6.5, y3+0.30, color='#B8860B')

# ── Row 4: L_anat (centre) ───────────────────────────────────
y4 = 1.65
box(ax2, 4.25, y4, 5.2, 0.75,
    'L_anat  =  (1/HW) · ‖S(G(x_A)) − S(x_A)‖²_F',
    fc='#FFE0B2', ec='#E65100', fontsize=10, bold=True)

arr(ax2, 2.0+0.75, y3-0.05, 4.25-1.2, y4+0.25, color='#E65100')
arr(ax2, 6.5-0.75, y3-0.05, 4.25+1.2, y4+0.25, color='#E65100')

# ── Row 5: gradient back to G only ───────────────────────────
y5 = 0.65
ax2.annotate('', xy=(4.0, y1-0.30), xytext=(4.0, y5+0.25),
             arrowprops=dict(arrowstyle='->', color='#1A7A40',
                             lw=2.0, linestyle='--',
                             connectionstyle='arc3,rad=0.3'),
             zorder=5)
ax2.text(5.55, 1.0, '∂L_anat / ∂G  (gradient to G only,\nSegmentor S stays frozen)',
         ha='left', va='center', fontsize=8.5, color='#1A7A40',
         bbox=dict(boxstyle='round,pad=0.25', fc='#E8F8EE',
                   ec='#1A7A40', lw=1.2))

# ── Annotations / No-gradient cross on S ─────────────────────
for xs in [2.0, 6.5]:
    ax2.text(xs, y2-0.55, '✗  no backprop', ha='center', fontsize=7.5,
             color='#C00060', style='italic')

ax2.set_title('AnatomyPreservingLoss — Detailed Computation Graph',
              fontsize=12, fontweight='bold', pad=8)

out2 = os.path.join(OUT_DIR, 'fig_flowchart2.png')
fig2.savefig(out2, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig2)
print(f"Saved: {out2}")

print("Done. Both flowcharts created.")
