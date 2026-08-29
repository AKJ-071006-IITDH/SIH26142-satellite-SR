# SIH26142 - Super Resolution Mapping of Satellite Imagery

This repository contains the end-to-end deep learning pipeline for super-resolution mapping (SRM) of Sentinel-style satellite imagery. The project upgrades low/medium resolution multispectral imagery into higher-resolution outputs, while also estimating uncertainty and exposing the system through a browser-based web dashboard.

The core training code lives in the project root and the deployable web app lives under `WEB/WEB`.

## Model status

- Current best-performing model: Phase 1
- Public repository status: Phase 1 is the default model represented in the checked-in code and checkpoint setup
- Experimental stages: Phase 2, Phase 3, and Phase 5 are additional research/benchmarking experiments and not the default stable model path for this repo

> Important: The code currently published in this GitHub repository reflects the best-performing model in the project at this stage: the Phase 1 model. This means the training and inference implementation in the main branch is tuned around the Phase 1 checkpoint and configuration, while Phase 2, Phase 3, and Phase 5 remain as additional experimental/model-development stages for benchmarking and research.

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
- Best measured image quality in the project
- Good for reliable, stable super-resolution output

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

### Loss functions used during training

- Charbonnier pixel loss: robust, smooth L1-like reconstruction loss
- Spectral consistency loss: compares NDVI between prediction and target
- Perceptual loss: can use VGG-based feature matching or self-feature matching
- Adversarial GAN objective: used in Phase 3 to enhance realism

### Uncertainty estimation

- Monte Carlo dropout is used during inference
- Multiple stochastic forward passes produce a variance/uncertainty map
- The web UI converts this to a confidence/uncertainty heatmap

### Evaluation metrics

The project measures:

- PSNR
- SSIM
- SAM (Spectral Angle Mapper)
- ERGAS

These are used to compare model performance on test patches.

---

## Repository structure

```text
sih26142-srm/
├── README.md                       # Project overview and deployment instructions
├── requirements.txt               # Python dependencies for training + app
├── src/                           # Core training and model logic
│   ├── data/                      # Data fetching, patch creation, dataset utilities
│   ├── eval/                      # Metric evaluation scripts
│   ├── losses/                    # Loss functions and NDVI logic
│   ├── models/                    # RRDBNet and model architecture code
│   ├── uncertainity/              # MC-dropout uncertainty logic
│   ├── train.py                   # Standard training entry point
│   ├── train_phase4.py            # Adversarial GAN training stage
│   ├── train_phase5.py            # Weight interpolation stage
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

---

## Evaluation

After training, evaluate metrics on a checkpoint:

```powershell
python -m src.eval.evaluate --checkpoint checkpoints/best_model_phase4.pt
```

You can also evaluate the test split explicitly:

```powershell
python -m src.eval.evaluate --checkpoint checkpoints/best_model_phase5.pt --split test
```

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
