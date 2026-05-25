# AP-CUT: Anatomy-Preserving Contrastive Unpaired Translation for Handheld Ultrasound Domain Adaptation

**Karrar M. Khudhair · Bareq M. Khudhair**  
Department of Computer TechniquesEngineeringImam Al-Kadhim University College(IKC), Iraq

---

## Overview

Handheld point-of-care ultrasound (POCUS) devices produce lower-quality images than high-end clinical systems, causing deep learning diagnostic models to fail when deployed on portable hardware. **AP-CUT** bridges this quality gap without retraining the downstream classifier.

Built on top of [CUT (Park et al., ECCV 2020)](https://github.com/taesungp/contrastive-unpaired-translation), AP-CUT introduces a novel **AnatomyPreservingLoss** that prevents the generator from distorting lesion morphology during domain translation:

```
L_anat = (1 / H·W) · ‖S(G(x_A)) − S(x_A)‖²_F
```

where `S` is a **frozen** UNet segmentor. Gradients flow only to the generator `G` — the segmentor is never updated.

---

## Results (BUSI dataset, seed=42, N=780)

| Condition | Accuracy | Macro-F1 | SSIM | PSNR |
|-----------|----------|----------|------|------|
| Degraded baseline | 52.05% | 0.4001 | — | — |
| AP-CUT + Random Segmentor | 80.00% | 0.7649 | — | — |
| **AP-CUT (Ours)** | **83.08%** | **0.8006** | **0.7057** | **24.22 dB** |

**Per-class AUC:** Normal = 0.946 · Benign = 0.931 · Malignant = 0.931

---

## Architecture

```
Domain A (degraded) ──► Generator G ──► Fake B
        │                                  │
        └──► Frozen Segmentor S ◄──────────┘
                     │
              L_anat (MSE of masks)
                     │
              ▼ gradient to G only
```

- **Generator:** ResNet-9blocks with instance normalisation
- **Discriminator:** PatchGAN 70×70 (LSGAN objective)
- **Loss:** `L_total = λ_GAN·L_GAN + λ_NCE·L_NCE + λ_anat·L_anat`  
  with `λ_GAN=1.0`, `λ_NCE=1.0`, `λ_anat=10.0`

---

## Project Structure

```
├── src/
│   ├── models/
│   │   ├── cut_model.py          # AP-CUT main model
│   │   ├── cyclegan_model.py     # CycleGAN baseline
│   │   ├── networks.py           # Generator, Discriminator
│   │   ├── patchnce.py           # PatchNCE contrastive loss
│   │   └── diagnostic_model.py   # Frozen ResNet-50 classifier
│   ├── losses/
│   │   ├── anatomy.py            # AnatomyPreservingLoss ← novel component
│   │   └── adversarial.py        # LSGAN loss
│   ├── data/
│   │   ├── dataset.py
│   │   ├── degradation.py        # Simulated handheld US degradation
│   │   └── transforms.py
│   ├── training/
│   │   └── trainer.py
│   └── evaluation/
│       └── evaluate.py
├── scripts/
│   ├── train_cut.py              # Train AP-CUT
│   ├── train_segmentor.py        # Pre-train frozen UNet segmentor
│   ├── train_diagnostic.py       # Fine-tune ResNet-50 diagnostic model
│   ├── run_evaluation.py         # Full evaluation pipeline
│   └── visualize_results.py
├── config/
│   ├── cut.yaml
│   └── train.yaml
├── figures/                      # Generated result figures
└── requirements.txt
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/bareqmaher-arch/Antomy-Presserving-Ultrasound.git
cd Antomy-Presserving-Ultrasound
pip install -r requirements.txt
```

### 2. Download dataset

Download the [BUSI dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) and place it in `data/raw/`:

```
data/raw/
├── benign/
├── malignant/
└── normal/
```

Then run:

```bash
python scripts/prepare_data.py
```

### 3. Train

```bash
# Step 1 — Train the UNet segmentor (used as frozen anatomy supervisor)
python scripts/train_segmentor.py

# Step 2 — Fine-tune the diagnostic model on clean images
python scripts/train_diagnostic.py

# Step 3 — Train AP-CUT
python scripts/train_cut.py --config config/cut.yaml
```

### 4. Evaluate

```bash
python scripts/run_evaluation.py
```

---

## Hardware

All experiments were conducted on an **NVIDIA RTX 3060 12 GB** with Python 3.11 and PyTorch 2.1. Mixed-precision training (`torch.cuda.amp`) is enabled by default.

---

## Key Files

| File | Description |
|------|-------------|
| `src/losses/anatomy.py` | **AnatomyPreservingLoss** — the novel contribution |
| `src/models/cut_model.py` | Full AP-CUT training loop |
| `src/data/degradation.py` | Handheld US degradation pipeline |
| `scripts/run_evaluation.py` | Reproduces all reported metrics |

---

## Citation

If you use this code, please cite:

```bibtex
@article{khudhair2026apcut,
  title   = {Anatomy-Preserving Contrastive Unpaired Translation for
             Handheld Ultrasound Domain Adaptation},
  author  = {Khudhair, Karrar M. and Khudhair, Bareq M.},
  year    = {2026},
  institution = {Department of Computer TechniquesEngineeringImam Al-Kadhim University College(IKC), Iraq}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
