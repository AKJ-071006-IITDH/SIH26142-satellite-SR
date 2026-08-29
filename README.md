# SIH26142 — Super Resolution Mapping of Satellite Imagery

An end-to-end, hardware-optimized Deep Learning system for Super-Resolution Mapping (SRM) of satellite imagery (Sentinel-2 L2A). Built for Smart India Hackathon (SIH26142 / NTRO).

## Key Features

1. **Input → Output Super-Resolution Pipeline**: Upgrades medium-resolution satellite imagery (30m/40m) to native 10m high-resolution outputs using an RRDBNet generator.
2. **Spectral & Geographic Consistency**: Loss function incorporates **Charbonnier Pixel Loss**, **Spectral NDVI Consistency Loss**, and **Feature Representation Loss** to prevent hallucinations.
3. **Explicit Uncertainty Management**: **Monte Carlo Dropout** inference (N stochastic passes) computes pixel-wise standard deviation, producing an **Inferno Uncertainty Heatmap** highlighting inferred vs directly observed features.
4. **Remote Sensing Evaluation**: Calculates **PSNR**, **SSIM**, **SAM (Spectral Angle Mapper)**, and **ERGAS**.
5. **Persistent Upload History**: Automatically saves uploaded and processed images, parameters, thumbnails, and metrics into a persistent SQLite store. Past runs can be re-loaded into the inspector at any time.
6. **Interactive Web Studio**: Glassmorphism web dashboard featuring an interactive curtain split-viewer, view mode switches (RGB, Uncertainty, NDVI), and metric counters.

---

## Hardware Optimization (NVIDIA RTX 4050 6GB VRAM)

Tuned for execution without Out-Of-Memory (OOM) errors:
- `num_blocks=8` (RRDBNet)
- `batch_size=4` with `accumulation_steps=4` (effective batch size 16)
- Automatic Mixed Precision (`torch.cuda.amp.autocast`)
- `num_workers=2`

---

## Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Pre-Demo Health Check
```bash
python demo/smoke_test.py
```

### 3. Launch Web Application Dashboard
```bash
python -m src.server
```
Open browser at: `http://localhost:8000`

### 4. Train RRDBNet Model
```bash
python -m src.train
```

### 5. Evaluate Test Metrics
```bash
python -m src.eval.evaluate
```

---

## Project Structure

```
WEB/
├── configs/
│   └── config.yaml             # Hyperparameters & settings
├── data/
│   ├── raw_tiles/              # Sentinel-2 & simulated raw tiles
│   ├── patches/                # Patch dataset
│   ├── splits/                 # Train/Val/Test JSON splits
│   └── history/                # Saved persistent history
├── src/
│   ├── data/                   # Fetch, patchify, degrade, dataset
│   ├── models/                 # RRDBNet with 2D Spatial Dropout
│   ├── losses/                 # Spectral NDVI & Charbonnier losses
│   ├── uncertainty/            # MC-Dropout engine & heatmaps
│   ├── eval/                   # PSNR, SSIM, SAM, ERGAS metrics
│   ├── db/                     # SQLite persistent history store
│   ├── server.py               # FastAPI backend web server
│   └── train.py                # Mixed precision training loop
├── frontend/                   # Interactive Glassmorphism Web App
│   ├── index.html
│   ├── styles.css
│   └── app.js
└── demo/                       # Health check & Streamlit fallback
    ├── app.py
    └── smoke_test.py
```
