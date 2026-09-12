# SIH26142 - Super Resolution Mapping of Satellite Imagery

This repository contains the end-to-end deep learning pipeline for super-resolution mapping (SRM) of Sentinel-style satellite imagery. The project upgrades low/medium resolution multispectral imagery into higher-resolution outputs, while also estimating uncertainty and exposing the system through a browser-based web dashboard.

The core training code lives in the project root and the deployable web app lives under `WEB/WEB`.

## Model status

**There are two current models, not one, because two different things are being optimised.**

| use case | model | checkpoint | headline |
|---|---|---|---|
| Analysis, band-ratio products, anything quantitative | **Attention-RRDB** | `best_model_attn.pt` | **37.73 dB / 0.889 SSIM** — best on every distortion metric |
| Imagery people look at | **Attention-RRDB + GAN v3** | `best_model_attn_gan_v3.pt` | **LPIPS 0.109 / HF 0.780** — half the perceptual distance to ground truth |

Neither dominates. PSNR/SSIM/SAM/ERGAS measure whether each pixel holds the right *value*; LPIPS and the HF energy ratio measure whether the image *looks* like the target. Past a point those objectives oppose each other — the perception–distortion tradeoff — so both checkpoints ship. See [Experiments and results](#experiments-and-results) for the full picture.

- The RRDBNet phase track (Phase 1/2/3/5) is the original model family and remains for reference.
- Attention-RRDB is the current architecture, with fidelity, pixel-weighted, adversarial and data-pipeline variants documented below.

> **Two caveats that affect every number in this README.**
>
> 1. The dataset was expanded partway through (9 regions × 3 seasons → 9 regions × 6 seasons, 1,469 → 3,264 patches). Metrics from before and after are not comparable.
> 2. The original `splits.py` shuffles at the *patch* level, so crops from one tile scatter across train/val/test. Measured: **all 54 tiles contribute to all three splits, and 326/326 test patches have a train patch directly adjacent**. A leak-free spatial split (`src/data/spatial_splits.py`) was added; headline numbers now come from it. The leak was worth about **0.2 dB** in a controlled comparison — worth fixing for defensibility, but it was not concealing a broken result.

## Overview

The system is designed to map medium-resolution satellite inputs (for example, 30m/40m bands) to a sharper 10m output using a custom RRDB-based generator. It is trained on 4-band imagery: Red, Green, Blue, and NIR, and includes uncertainty estimation via Monte Carlo dropout.

The workflow is:

1. Prepare or fetch satellite tiles and patch data.
2. Train a super-resolution model on low-resolution / high-resolution patch pairs.
3. Evaluate image quality using PSNR, SSIM, SAM, and ERGAS.
4. Run the model through a FastAPI backend and a frontend dashboard.
5. Upload imagery or pick sample tiles and inspect the output, uncertainty map, and NDVI view.

---

## What the model does

The project uses a residual-dense generator architecture based on ESRGAN/RRDBNet ideas.

### Main model family: RRDBNet

- Input: 4-channel image tensor in order `[R, G, B, NIR]`
- Output: 4-channel super-resolved image
- Scaling: `x4` upsampling
- Backbone: residual-in-residual dense blocks (RRDB)
- Improvement: channel attention and dropout are used to stabilize training and estimate uncertainty

### Model variants used in this project

#### Phase 1 - Refined Baseline
- 6-block RRDBNet
- Was the best model in the project before the Attention-RRDB track; now superseded
- Still used as the frozen feature extractor for the fidelity track's perceptual loss

#### Phase 2 - High-Capacity Model
- 12-block RRDBNet
- Higher capacity for fine spatial detail
- Stronger detail potential, but not always better metrics than Phase 1

#### Phase 3 - Adversarial Refinement
- Warm-started from Phase 2
- Uses a PatchGAN discriminator for sharper texture and more realistic local detail
- Good for visual fidelity refinement

#### Phase 5 - Weight Interpolation Blend
- Combines Phase 2 and Phase 3 weights
- Intended to keep the strong structure of Phase 2 and the sharpness of Phase 3
- Used as the final blended checkpoint when desired

### Attention-RRDB track (current architecture)

A separate generator family (`src/models/rrdbnet_attn.py`, `AttentionRRDBNet`), kept in its own set of files so it stays independently readable/diffable from the original RRDBNet track above. Same overall shape (shallow conv -> RRDB stack -> upsample -> output conv) plus two changes aimed at getting closer to ground truth on a modest dataset:

- **CBAM attention** (channel + spatial) inside every dense block, so the network reweights *which* features/locations matter instead of relying on raw depth alone.
- **Image-space long skip**: the output is a bicubic-upsampled copy of the input plus a learned residual, with the residual's output conv zero-initialized so training starts as *exactly* bicubic upsampling and only has to learn the remaining detail -- much easier to fit well from a few thousand patches than reconstructing the whole image from scratch.

Two training configurations exist over this same architecture:

- **`src/train_v2.py`** -- the primary, best-performing run (`checkpoints/best_model_attn.pt`). Loss weights: pixel 0.8 / SSIM 0.3 / spectral (NDVI) 0.5 / perceptual 0.3, dropout 0.1.
- **`src/train_v2_fidelity.py`** -- an experiment shifting the loss further toward raw pixel accuracy (pixel 1.0 / SSIM 0.2 / spectral 0.2 / perceptual 0.3, dropout 0.05) on the theory that more pixel-loss weight would improve fidelity metrics further. Measured result: it didn't — see **Experiment 1** under Experiments and results below. Kept as a documented negative result in its own checkpoint rather than deleted.

Both write to distinct checkpoint filenames (`*_attn.pt` / `*_attn_fidelity.pt`) so neither run can overwrite the other, and both train purely on fidelity terms -- no adversarial/GAN loss.

### Loss functions used during training

| term | definition | used by |
|---|---|---|
| **Charbonnier** (pixel) | √(d² + ε²), ε = 1e-3 — smooth L1 with bounded gradients, so outlier pixels can't dominate | all tracks |
| **NDVI L1** (spectral) | (NIR − Red) / (NIR + Red + 1e-2) — what makes this remote-sensing-aware rather than generic photo SR | all tracks |
| **SSIM** (structural) | 1 − mean(SSIM), 11×11 Gaussian σ=1.5, C1=1e-4, C2=9e-4 | Attention-RRDB tracks |
| **Perceptual** (feature) | fidelity track: L1 on frozen RRDBNet features. Adversarial track: **L1 on frozen VGG19 conv5_4** | all tracks |
| **Adversarial** (realism) | BCE-with-logits against a PatchDiscriminator, weight 0.005, label smoothing 0.9/0.0 | adversarial track only |

Two implementation notes that cost real debugging time:

- **The perceptual extractor must be a separate model instance.** `SpectralPerceptualLoss` sets `requires_grad=False` on whatever it is handed, and `nn.Module` attributes are references — so passing the live generator silently freezes its `conv_first` and early RRDB blocks. This bug is present in `train_phase4.py` (46% of the generator frozen) and very likely explains why the Phase 3/4 results disappointed.
- **The RRDBNet-based perceptual loss is weak** because those features were themselves trained with fidelity losses, so they encode the same blur-tolerance and cannot object to blur the pixel loss already accepts. VGG19 was trained discriminatively and does.

### Uncertainty estimation

- Monte Carlo dropout is used during inference
- Multiple stochastic forward passes produce a variance/uncertainty map
- The web UI converts this to a confidence/uncertainty heatmap

### Evaluation metrics

**Distortion metrics** — does each pixel hold the right value?

- **PSNR**, **SSIM** — higher is better
- **SAM** (Spectral Angle Mapper) — band-relationship accuracy, lower is better
- **ERGAS** — lower is better

These four have a blind spot that matters here: a prediction that hedges toward the local average scores *well* on them, because hedging minimises squared error. That is why a model can sit at 37+ dB and still look visibly smoother than the ground truth.

**Perceptual metrics** (`src/eval/perceptual_metrics.py`) — does the image *look* like the target? Both are still reference-based, i.e. compared against the real HR crop.

- **LPIPS** — deep-feature distance, lower is better. Uses an **AlexNet** backbone deliberately: the GAN trains against VGG19, so scoring with VGG would be circular. Requires `pip install lpips`.
- **HF energy ratio** — high-pass energy of the prediction ÷ that of the target. **1.0 = exactly as much fine detail as the truth**; below 1.0 = blurrier. No dependencies, and it names the failure in a number that explains itself. Read alongside LPIPS, never alone: bicubic scores 0.563 largely by propagating input noise, and it cannot separate real detail from the PixelShuffle mesh.

**Artifact measurement** (`src/eval/checkerboard.py`) — Fourier analysis of smooth patches, reporting power at the period-2 and period-4 frequencies where PixelShuffle artifacts live, relative to ground truth.

---

## Experiments and results

Every experiment below is a **single-variable change** against a named baseline, kept in its own script and writing to its own checkpoint so nothing overwrites anything. Negative results are documented rather than deleted — several of them redirected the work more usefully than the wins did.

### Full results — leak-free test split (408 crops, all 9 regions)

The numbers to quote. No test crop touches a training crop.

| model | PSNR ↑ | SSIM ↑ | SAM ↓ | ERGAS ↓ | LPIPS ↓ | HF → 1.0 |
|---|---|---|---|---|---|---|
| bicubic baseline | 33.772 | 0.8041 | 5.409 | 5.753 | 0.3854 | 0.563 |
| RRDBNet Phase 2 | 37.183 | 0.8811 | 3.078 | 3.888 | 0.2508 | 0.519 |
| **attn** | **37.734** | **0.8888** | **2.873** | **3.685** | 0.2219 | 0.532 |
| attn_fidelity | 37.442 | 0.8857 | 3.009 | 3.779 | 0.2329 | 0.533 |
| attn_v3 *(confounded)* | 37.345 | 0.8850 | 3.043 | 3.810 | 0.2422 | 0.540 |
| attn_gan | 36.506 | 0.8619 | 3.377 | 4.189 | 0.1189 | 0.703 |
| attn_gan_v2 | 36.403 | 0.8638 | 3.317 | 4.081 | 0.1423 | 0.697 |
| **attn_gan_v3** | 36.087 | 0.8508 | 3.515 | 4.333 | **0.1091** | **0.780** |

Measured with `python -m src.eval.perceptual_metrics --manifest data/splits/manifest_spatial.json --compare`.

### Original patch-level split (326 patches) — kept for continuity

Every decision made before the split audit was scored against this. **Read it as optimistic.**

| model | PSNR ↑ | SSIM ↑ | SAM ↓ | ERGAS ↓ | LPIPS ↓ | HF → 1.0 |
|---|---|---|---|---|---|---|
| bicubic baseline | 33.503 | 0.7922 | 5.259 | 5.763 | 0.3808 | 0.546 |
| RRDBNet Phase 2 | 36.816 | 0.8693 | 3.049 | 3.883 | 0.2580 | 0.504 |
| attn | 37.257 | 0.8779 | 2.884 | 3.699 | 0.2303 | 0.520 |
| attn_fidelity | 37.019 | 0.8750 | 3.008 | 3.789 | 0.2405 | 0.519 |
| attn_gan | 36.098 | 0.8515 | 3.395 | 4.219 | 0.1230 | 0.698 |
| attn_gan_v2 | 35.989 | 0.8530 | 3.356 | 4.113 | 0.1442 | 0.693 |

---

### Experiment 1 — loss reweighting (`train_v2_fidelity.py`)

**Hypothesis.** Late in `attn`'s training, the pixel term accounted for only 13% of the remaining loss while SSIM and spectral took 82%. Shifting weight toward pixel accuracy should improve PSNR.

**Change.** pixel 0.8→1.0, SSIM 0.3→0.2, spectral 0.5→0.2, dropout 0.1→0.05.

**Result: worse on every metric** — 37.44 dB vs 37.73. PSNR, SSIM, SAM and ERGAS are themselves structural and spectral measures, so de-weighting the terms that optimise structure and spectral consistency traded away exactly what the metrics reward.

**The more useful output was a stability finding.** This config's gradients measured **2.2× larger** than `attn`'s (median norm 0.254 vs 0.114), leaving only 1.45× headroom below float16's overflow ceiling versus 2.5×. Under fp16 it overflowed repeatedly, collapsed the loss scale, and produced NaN in **all 396 weight tensors**. Switching to **bfloat16** — which carries fp32's exponent range and needs no `GradScaler` — fixed it outright, and the fix was applied to every training script.

*A third variant (`train_v2_fidelity_2.py`) probing the opposite direction (pixel 0.6, SSIM 0.4, spectral 0.6) is available but was not run to completion.*

### Experiment 2 — adversarial refinement (`train_v2_gan.py`)

**Why.** Every term in the fidelity loss is minimised by predicting the conditional mean. Where the model can't tell a roof edge from a shadow, the lowest-error answer is something between — and averaged over a patch, that is blur. `attn` and `attn_fidelity` land 0.29 dB apart because reweighting cannot escape a property they share. A discriminator can: a blurry patch is trivially identifiable as fake regardless of its MSE.

**Setup.** Generator warm-started from `best_model_attn.pt`, architecture unchanged (6,158,028 params). Adds a `PatchDiscriminator` (559,361 params, emitting a 1×32×32 verdict map rather than one score) and swaps the self-feature perceptual loss for **frozen VGG19 conv5_4** — the self-feature version was trained with fidelity losses itself, so it encoded the same blur-tolerance and could not object to blur the pixel loss already accepted. Checkpoints are selected on **LPIPS, never fidelity loss**, because fidelity loss *rises* during adversarial training and would pick the blurriest model.

**Three runs:**

| run | `w_adv` | `lr_d` | EMA | outcome |
|---|---|---|---|---|
| attn_gan (v1) | 5e-3 | 1e-4 | no | best at **epoch 3 of 40**; D loss collapsed to 0.19 by epoch 9, spectral term spiked 2.5×, PSNR fell to 33.9 |
| attn_gan_v2 | 3e-3 | 2e-5 | no | stable 15 epochs but under-sharpened — two variables changed at once |
| **attn_gan_v3** | **5e-3** | **2e-5** | **0.999** | **stable all 40 epochs**, D loss held 0.35–0.46, spectral flat |

#### attn_gan_v3 — the run that actually trained

**The decisive fact about v1 was easy to miss: its best checkpoint arrived at epoch 3 of 40.** It was never a converged model, just one caught mid-collapse. v2 then changed *two* things at once — slower discriminator **and** weaker adversarial weight — and the weaker weight is what cost it the sharpness. That left exactly one corner of the grid untried: **v1's adversarial pressure with v2's slow discriminator.**

**Change 1 — `w_adv` 5e-3 (v1's), `lr_d` 2e-5 (v2's).** Keep the setting that produced the sharpness; change only the one that caused the collapse.

**Change 2 — generator EMA, decay 0.999.** Adversarial training does not converge, it oscillates: v1's LPIPS ran 0.119 → 0.131 → 0.123 → 0.133 on consecutive epochs, so which checkpoint you keep is partly luck of the draw. An exponential moving average of the weights sits nearer the centre of that oscillation than any single point on it. Scoring and checkpointing use the averaged weights; training continues on the live ones. Implemented as the `EMA` class in `train_v2_gan.py`, enabled by default via `ema_decay=0.999`.

**What the run did.** Stable for all 40 epochs, and still improving at the end:

| | epoch 1 | epoch 10 | epoch 20 | epoch 40 | best (ep 38) |
|---|---|---|---|---|---|
| LPIPS ↓ | 0.2108 | 0.1437 | 0.1206 | 0.1091 | **0.1089** |
| HF ratio | 0.533 | 0.631 | 0.704 | 0.811 | 0.804 |
| PSNR | 37.780 | 37.098 | 36.569 | 36.037 | 36.064 |
| D loss | 0.623 | 0.411 | 0.453 | 0.450 | — |

Three health signals, all of which v1 failed:
- **D loss never entered collapse** — held 0.35–0.46 the whole run, against v1 hitting 0.19 by epoch 9.
- **Spectral term stayed flat** — 0.0136 → 0.0157, against v1's 2.5× spike to 0.0403.
- **LPIPS descended monotonically** instead of thrashing, which is the EMA working as intended.

**Result on the leak-free split:**

| | attn (baseline) | attn_gan (v1) | **attn_gan_v3** |
|---|---|---|---|
| LPIPS ↓ | 0.2219 | 0.1189 | **0.1091** |
| HF → 1.0 | 0.532 | 0.703 | **0.780** |
| PSNR | 37.734 | 36.506 | 36.087 |

**vs the fidelity baseline: LPIPS −51%, HF +47%, for 1.65 dB of PSNR.** vs v1: 8% better LPIPS and 11% more sharpness for a further 0.42 dB. The entire gain came from letting the run train — same architecture, same loss weights, one hyperparameter moved and an EMA added.

**The honest cost:** SAM moved 2.873 → 3.515, worse than even Phase 2. The adversarial term measurably distorts band relationships. **NDVI or any band-ratio product derived from `attn_gan_v3` is less trustworthy than from `attn`** — which is precisely why both checkpoints ship rather than one replacing the other.

**Reproduce it:**

```powershell
python -c "from src.train_v2_gan import train; train(w_adv=5e-3, lr_d=2e-5, num_epochs=40, run_suffix='_v3')"
```

### Experiment 3 — leak-free splits and random crops (`train_v3.py`)

**The finding.** Auditing `splits.py` showed all 54 tiles contributing to all three splits and 326/326 test patches having a directly adjacent train patch — same field, same day, same illumination.

**The fix.** `src/data/spatial_splits.py` offers two modes:
- **spatial** (default) — each tile is carved into disjoint bands along its longer axis with a **64 px guard** between them. All 9 regions and 6 seasons appear in train, val *and* test.
- **region** — whole regions held out; test geography entirely unseen. Stricter, but with only 9 regions it made val a single land-cover type (`goa_coastal`, the easiest region, ~13 dB above `chennai_urban`) — and since `val_loss` drives early stopping, that biases checkpoint selection. Retained as a secondary generalisation check only.

**The second change.** `src/data/tile_dataset.py` samples **random crops from memory-mapped raw tiles** instead of the 2,449 fixed pre-cut patches, so a run draws ~200,000 distinct crops from imagery already on disk.

**Result: inconclusive, not negative.** v3 came in 0.39 dB below `attn`, but the comparison is confounded against it — `attn` trained on crops from the *whole* of every tile including v3's test band, so it is scored on data it trained on. v3 also trained on 60% of each tile's area versus `attn`'s ~75%. Settling this needs a control trained on the same band with fixed patches.

### Experiment 4 — the checkerboard artifact (`src/eval/checkerboard.py`)

The adversarial outputs carry a faint regular mesh. The first hypothesis — discriminator dominance — was testable, and `attn_gan_v2` disproved it: far more stable, mesh unchanged. Fourier analysis of smooth patches then located the real cause:

| model | period-2 | period-4 |
|---|---|---|
| ground truth | 2.6× | 151.0× |
| attn *(never met a discriminator)* | 39.3× | **2978.4×** |
| attn_fidelity | 42.0× | 2911.6× |
| attn_gan | 603.9× | 417.0× |
| attn_gan_v2 | 788.1× | 625.3× |

**The checkerboard is architectural, not adversarial.** `PixelShuffle` builds each 2×2 output block from four *independently initialised* convolution kernels; they disagree from step zero and tile that disagreement across the image. Two ×2 stages place artifacts at period 4 (first stage, doubled by the second) and period 2 (second stage) — exactly the frequencies measured. Adversarial training doesn't create it, it *unmasks* it by rewarding the high-frequency energy that fidelity losses suppress.

**The fix** is **ICNR initialisation** — initialise those four kernels as identical copies so the upsampler starts as exact nearest-neighbour resizing with zero checkerboard (verified: within-2×2-block spread 2.775 → 0.000). Same reasoning as the zero-initialised `conv_last`: start from a known-good state. It only applies at initialisation, so it needs a retrain — wired as `src/train_v2_icnr.py` (stage 1) and `src/train_attn_gan_v3_icnr.py` (stage 2), not yet run to completion.

> **Caveat on the HF ratio:** it cannot distinguish recovered detail from the mesh, because the mesh *is* high-frequency energy. Part of `attn_gan_v3`'s 0.780 is artifact rather than real structure.

### Experiments that were measured and dropped

- **Degradation model / aliasing.** `degrade()` blurs at σ=1.0 before ×4 subsampling; proper anti-aliasing wants σ≈2. Measured: only **0.81%** of image energy sits above the Nyquist limit at σ=1.0, and fixing it buys ~0.3 dB round-trip. Not the lever it appeared to be.
- **A pixel-weighted variant in the opposite direction** (Experiment 1's mirror) — built, not run, after the first reweighting produced nothing.

### What actually moved the needle

| change | effect |
|---|---|
| CBAM attention + bicubic skip + capacity | +0.44 dB (≈0.12 dB once leak-corrected) |
| Loss reweighting (twice) | worse both times |
| Random crops / leak-free split | inconclusive, ≤0.4 dB either way |
| Degradation-model fix | ~0.3 dB |
| **Adversarial objective + stable training** | **LPIPS 0.222 → 0.109, HF 0.532 → 0.780** |

Every change on the distortion axis lands within ±0.4 dB. The one change that transformed the output was to the *objective*.

---

## Repository structure

```text
sih26142-srm/
├── README.md                       # Project overview and deployment instructions
├── requirements.txt               # Python dependencies for training + app
├── src/                           # Core training and model logic
│   ├── data/                      # Data fetching, patch creation, dataset utilities
│   │   ├── fetch.py               # Sentinel-2 L2A via CDSE (skips tiles already on disk)
│   │   ├── patchify.py            # Tiles -> 128x128 patches
│   │   ├── degrade.py             # HR -> LR: blur, subsample, add sensor noise
│   │   ├── splits.py              # ORIGINAL patch-level split (leaky -- see Experiment 3)
│   │   ├── spatial_splits.py      # Leak-free split: spatial bands or region holdout
│   │   └── tile_dataset.py        # Random crops from memory-mapped raw tiles
│   ├── eval/                      # Metric evaluation scripts
│   │   ├── evaluate.py            # Metrics for the RRDBNet phase track
│   │   ├── evaluate_v2.py         # Metrics for the Attention-RRDB track
│   │   ├── perceptual_metrics.py  # LPIPS + HF energy ratio + all six, --compare
│   │   ├── checkerboard.py        # Fourier analysis of the PixelShuffle artifact
│   │   ├── visualize.py           # Side-by-side grid (auto-detects architecture)
│   │   ├── visualize_all.py       # Per-patch comparison images
│   │   ├── compare_models.py      # ALL models side by side on the same crops
│   │   └── plot_history.py        # Train-vs-val loss curve plotting
│   ├── losses/                    # Loss functions and NDVI logic
│   │   ├── losses.py              # Charbonnier + spectral + perceptual
│   │   ├── losses_attn.py         # Adds SSIM, composes CombinedFidelityLossV2
│   │   └── losses_gan.py          # VGG19 perceptual + generator/discriminator objectives
│   ├── models/                    # Model architecture code
│   │   ├── rrdb.py                # Original RRDBNet generator
│   │   ├── discriminator.py       # PatchDiscriminator (spectral norm)
│   │   └── rrdbnet_attn.py        # AttentionRRDBNet: CBAM + bicubic skip + ICNR option
│   ├── uncertainity/              # MC-dropout uncertainty logic
│   ├── train.py                   # RRDBNet track: standard training entry point
│   ├── train_phase4.py            # RRDBNet track: adversarial stage (has the frozen-generator bug)
│   ├── train_phase5.py            # RRDBNet track: weight interpolation stage
│   ├── train_v2.py                # Attention-RRDB: primary fidelity run
│   ├── train_v2_fidelity.py       # Experiment 1: pixel-weighted loss
│   ├── train_v2_fidelity_2.py     # Experiment 1 mirror: structure-weighted (not run)
│   ├── train_v2_gan.py            # Experiment 2: adversarial refinement (+ EMA)
│   ├── train_v3.py                # Experiment 3: random crops on the leak-free split
│   ├── train_v2_icnr.py           # Experiment 4 stage 1: ICNR base retrain
│   ├── train_attn_gan_v3_icnr.py  # Experiment 4 stage 2: GAN on the ICNR base
│   └── ...
├── configs/
│   └── config.yaml                # Config for models, checkpoints, training, server
├── checkpoints/                   # Trained model checkpoints (.pt)
├── data/                          # Dataset, patches, splits, history storage
├── demo/                          # Smoke tests / demo scripts
├── WEB/
│   └── WEB/                       # Web application project folder
│       ├── README.md
│       ├── requirements.txt
│       ├── frontend/              # Static frontend (HTML/CSS/JS)
│       ├── src/                   # FastAPI backend
│       ├── configs/
│       ├── checkpoints/
│       └── data/
├── notebooks/
├── scripts/
├── TrainingLogs/
└── ...
```

---

## Requirements

### Recommended environment

- Python 3.10 or newer
- CUDA-capable NVIDIA GPU strongly recommended
- Windows, Linux, or WSL2 supported

### Python dependencies

Install project dependencies from the root or from the web project folder:

```powershell
pip install -r requirements.txt
```

If you are running the web app from `WEB/WEB`, use:

```powershell
cd WEB\WEB
pip install -r requirements.txt
```

### Optional GPU install for PyTorch

If you have an NVIDIA GPU, install the CUDA build of PyTorch instead of the CPU build:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Then verify:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Local setup

### 1) Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

### 3) Run the web application

The web app is served by FastAPI and uses the static frontend under `WEB/WEB/frontend`.

From the project root:

```powershell
cd WEB\WEB
python -m src.server
```

Then open in the browser:

```text
http://localhost:8000
```

The backend exposes the API and serves the UI from the same port.

### 4) Deploy the web part in production style

For a direct server deployment, you can start the app with Uvicorn directly:

```powershell
cd WEB\WEB
uvicorn src.server:app --host 0.0.0.0 --port 8000
```

This is the recommended command if you want to run it as a deployment service or behind a reverse proxy.

If you are using a process manager or Docker, the same app can be launched with:

```powershell
cd WEB\WEB
python -m src.server
```

---

## Running training

From the repository root, you can train the main model:

```powershell
python -m src.train
```

For adversarial GAN refinement:

```powershell
python -m src.train_phase4
```

For the final interpolated model blend:

```powershell
python -m src.train_phase5 --phase3 checkpoints/best_model_phase3.pt --phase4 checkpoints/best_model_phase4.pt --out checkpoints/best_model_phase5.pt
```

### Attention-RRDB track

```powershell
python -m src.train_v2               # primary fidelity run -> best_model_attn.pt
python -m src.train_v2_fidelity      # Experiment 1: pixel-weighted -> best_model_attn_fidelity.pt
```

### Adversarial track (best perceptual quality)

```powershell
# Experiment 2, the settings that worked: v1's adversarial pressure with v2's
# slow discriminator, plus generator EMA. -> best_model_attn_gan_v3.pt
python -c "from src.train_v2_gan import train; train(w_adv=5e-3, lr_d=2e-5, num_epochs=40, run_suffix='_v3')"
```

`run_suffix` is appended to every output filename, so re-runs with different hyperparameters cannot clobber a previous run's checkpoint — adversarial runs are volatile and the good checkpoint often arrives early.

### Leak-free data pipeline (Experiment 3)

```powershell
python -m src.data.spatial_splits --mode spatial --out data/splits/manifest_spatial.json
python -m src.data.spatial_splits --mode region  --out data/splits/manifest_region.json
python -m src.train_v3               # random crops on the leak-free split
```

### ICNR checkerboard fix (Experiment 4, two stages)

```powershell
python -m src.train_v2_icnr              # stage 1: ICNR base retrain  (~4h)
python -m src.train_attn_gan_v3_icnr     # stage 2: GAN on that base   (~1.5h)
```

ICNR only applies at *initialisation*, and the GAN stage overwrites the generator with a loaded checkpoint — so the base genuinely has to be retrained. To check ICNR survives training before committing four hours, run the 8-epoch A/B first:

```powershell
python -c "from src.train_v2_icnr import train; train(num_epochs=8, icnr_init=True,  run_suffix='_icnr_ab_on')"
python -c "from src.train_v2_icnr import train; train(num_epochs=8, icnr_init=False, run_suffix='_icnr_ab_off')"
python -m src.eval.checkerboard --compare
```

All training scripts use AMP (**bfloat16** where supported — see Experiment 1 for why fp16 was abandoned), a linear LR warmup, gradient-norm clipping with a non-finite guard, and early stopping on validation loss.

---

## Evaluation

After training, evaluate metrics on a checkpoint (RRDBNet phase track):

```powershell
python -m src.eval.evaluate --checkpoint checkpoints/best_model_phase4.pt
```

You can also evaluate the test split explicitly:

```powershell
python -m src.eval.evaluate --checkpoint checkpoints/best_model_phase5.pt --split test
```

For the Attention-RRDB track, use `evaluate_v2` instead:

```powershell
python -m src.eval.evaluate_v2 --checkpoint checkpoints/best_model_attn.pt --split test
```

### Full metrics table — all models, all six metrics

This is the one to use. `--manifest` switches to the leak-free split and is **required** for any model trained by `train_v3.py`:

```powershell
python -m src.eval.perceptual_metrics --manifest data/splits/manifest_spatial.json --compare
```

Needs `pip install lpips`; without it the LPIPS column reports `nan` and everything else still works.

### Checkerboard artifact measurement

```powershell
python -m src.eval.checkerboard --compare
```

### Visual evaluation

All visual tools auto-detect which architecture a checkpoint was trained with, and all accept `--manifest` to use the leak-free split:

```powershell
# Every model side by side on the same crops -- the most useful single view
python -m src.eval.compare_models --manifest data/splits/manifest_spatial.json --num_samples 6 --out demo/visual_eval_all.png

# Zoomed stress test: magnified crops of the highest-texture patches, where
# the difference between real detail and grain is actually legible
python -m src.eval.compare_models --manifest data/splits/manifest_spatial.json --crop 56 --hardest --out demo/visual_eval_zoom.png

# One model, original 4-column format
python -m src.eval.visualize --checkpoint checkpoints/best_model_attn_gan_v3.pt --num_samples 6 --out demo/visual_eval_attn_gan_v3.png

# One comparison image per test crop, across the whole split
python -m src.eval.visualize_all --checkpoint checkpoints/best_model_attn_gan_v3.pt --manifest data/splits/manifest_spatial.json --out_dir demo/full_test_attn_gan_v3
```

### Training curves

```powershell
python -m src.eval.plot_history --history checkpoints/training_history_attn.json
```

GAN histories additionally record per-epoch PSNR, SSIM, LPIPS, HF ratio and discriminator loss, which is what makes the perception–distortion tradeoff visible as it unfolds.

---

## Web app usage

Once the server is running:

1. Open `http://localhost:8000`
2. Select a sample tile or upload your own image
3. Choose a trained model from the model list
4. Set the Monte Carlo sample count
5. Run inference
6. Review:
   - low-resolution input
   - super-resolved output
   - ground truth / reference view
   - uncertainty map
   - NDVI view
   - evaluation metrics

The app stores run history in SQLite and gives access to previously generated results.

---

## API endpoints

The FastAPI backend exposes a few useful endpoints:

- `GET /api/status` → checks the server and device status
- `GET /api/models` → lists model IDs and availability
- `GET /api/tiles` → lists sample tiles from the dataset
- `POST /api/predict` → runs SR inference on a selected tile or uploaded file
- `GET /api/history` → lists saved inference history
- `GET /api/history/{entry_id}` → retrieves one entry
- `DELETE /api/history/{entry_id}` → removes one entry

---

## Key configuration file

The main runtime configuration is located here:

- `WEB/WEB/configs/config.yaml`

This file defines:

- dataset folders
- patch size and scale factor
- model registry entries
- checkpoint paths
- training parameters
- server port and host

---

## Recommended workflow

For a typical development cycle:

```powershell
# 1. Create environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train a model
python -m src.train

# 4. Run the app
cd WEB\WEB
python -m src.server
```

Then open the browser at `http://localhost:8000` and test the model visually.

---

## Notes

- The project is optimized for a CUDA-enabled workstation and can be run on CPU if needed, but GPU is strongly recommended for training and real-time inference.
- Checkpoint files must be present in `checkpoints/` before the web UI can load them.
- The root project and the web project are closely related; most users run training from the root and deploy the web app from `WEB/WEB`.

---

## Citation / project context

This project was developed for SIH26142 and focuses on applied remote sensing and deep learning for super-resolution mapping from satellite imagery.

Use this repository for:

- training high-quality SR models on multispectral patches
- generating uncertainty-aware super-resolution outputs
- deploying a lightweight web dashboard for visualization and evaluation
