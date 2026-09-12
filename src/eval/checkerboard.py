# src/eval/checkerboard.py
"""
Measures the pixel-shuffle checkerboard artifact by Fourier analysis.

PixelShuffle(r) builds each r x r output block from r^2 *different* convolution
kernels. Initialised independently, they disagree from step zero, and that
disagreement tiles across the image as a fixed periodic pattern. With two x2
stages the artifact lands at two specific frequencies: period 4 (the first
stage's pattern, doubled by the second) and period 2 (the second stage's).

This script measures power at exactly those frequencies, relative to each
image's own median power, and compares against the ground truth. Ground truth
sets the reference for how much periodic structure real imagery has; a large
excess over that is the artifact.

Measured on smooth patches on purpose -- checkerboard is only cleanly
separable where there is little real texture to hide it.

Read the numbers as a comparison between models, not as absolutes: the ratio
is normalised by each image's median power, so a model with little broadband
high-frequency content (a blurry one) gets a smaller denominator and an
inflated ratio.

Usage:
    python -m src.eval.checkerboard --compare
    python -m src.eval.checkerboard --checkpoint checkpoints/best_model_attn_icnr.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.data.dataset import SatelliteSRDataset
from src.eval.visualize import load_model

DEFAULT_CHECKPOINTS = [
    ("attn",          "checkpoints/best_model_attn.pt"),
    ("attn_fidelity", "checkpoints/best_model_attn_fidelity.pt"),
    ("attn_icnr",     "checkpoints/best_model_attn_icnr.pt"),
    ("icnr_ab_on",    "checkpoints/best_model_attn_icnr_ab_on.pt"),
    ("icnr_ab_off",   "checkpoints/best_model_attn_icnr_ab_off.pt"),
    ("attn_gan",      "checkpoints/best_model_attn_gan.pt"),
    ("attn_gan_v2",   "checkpoints/best_model_attn_gan_v2.pt"),
    ("attn_gan_icnr", "checkpoints/best_model_attn_gan_icnr.pt"),
]


def periodic_power(img):
    """Mean power spectrum of the high-passed image, fftshifted."""
    hp = img - F.avg_pool2d(img, 3, stride=1, padding=1)
    spec = torch.fft.fft2(hp.mean(1))       # collapse bands, then 2D FFT
    power = (spec.abs() ** 2).mean(0)       # average over batch
    return torch.fft.fftshift(power).numpy()


def period_peaks(power, size=128):
    """
    Power at the period-2 and period-4 frequencies, over the image median.

    After fftshift, index c is frequency 0 and c+k is frequency k. Period 2 is
    Nyquist (k = size/2), which exists only as the negative frequency -- hence
    the wrap-around indexing.
    """
    c = size // 2

    def at(y, x):
        return power[y % size, x % size]

    def ring(period):
        k = int(round(size / period))
        vals = []
        for d in range(-1, 2):
            vals.append(at(c + k, c + d))
            vals.append(at(c - k, c + d))
            vals.append(at(c + d, c + k))
            vals.append(at(c + d, c - k))
        return float(np.mean(vals))

    baseline = float(np.median(power))
    return ring(2) / baseline, ring(4) / baseline


def smooth_patch_loader(split="test", n=24, batch_size=8):
    dataset = SatelliteSRDataset(augment=False)
    with open(f"data/splits/{split}.json") as f:
        files = set(json.load(f))
    indices = [i for i, f in enumerate(dataset.patch_files) if str(f) in files]

    scored = []
    for i in indices[:: max(1, len(indices) // 120)]:
        _, hr = dataset[i]
        hp = hr - F.avg_pool2d(hr.unsqueeze(0), 3, 1, 1)[0]
        scored.append((hp.pow(2).mean().item(), i))
    scored.sort()
    smooth = [i for _, i in scored[:n]]
    return DataLoader(Subset(dataset, smooth), batch_size=batch_size,
                       shuffle=False, num_workers=0), len(smooth)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--split", default="test")
    parser.add_argument("--patches", type=int, default=24)
    args = parser.parse_args()

    device = "cpu"   # FFT on CPU is cheap and avoids device juggling
    loader, n = smooth_patch_loader(args.split, args.patches)
    print(f"Measuring periodic power on the {n} smoothest '{args.split}' patches.\n")

    targets = ([(n_, p) for n_, p in DEFAULT_CHECKPOINTS if Path(p).exists()]
               if args.compare else [("model", args.checkpoint)])

    rows = []
    # ground truth first -- it is the reference every model is judged against
    powers = []
    for _, hr_b in loader:
        powers.append(periodic_power(hr_b))
    rows.append(("ground truth", *period_peaks(np.mean(powers, axis=0))))

    for name, path in targets:
        model = load_model(path, device)
        powers = []
        with torch.no_grad():
            for lr_b, _ in loader:
                powers.append(periodic_power(model(lr_b).clamp(0, 1)))
        rows.append((name, *period_peaks(np.mean(powers, axis=0))))

    gt2, gt4 = rows[0][1], rows[0][2]
    hdr = f"{'model':<16}{'period-2':>12}{'period-4':>12}{'excess vs truth':>20}"
    print(hdr)
    print("-" * len(hdr))
    for name, p2, p4 in rows:
        excess = "" if name == "ground truth" else f"{p2/gt2:.0f}x / {p4/gt4:.0f}x"
        print(f"{name:<16}{p2:>11.1f}x{p4:>11.1f}x{excess:>20}")
    print("-" * len(hdr))
    print("Peaks are relative to each image's median power. Excess is over ground truth,")
    print("which is what a model with no checkerboard would match.")


if __name__ == "__main__":
    main()
