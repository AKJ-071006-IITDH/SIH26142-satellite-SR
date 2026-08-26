# src/eval/visualize.py
"""
Visual evaluation to sit alongside the numeric metrics from evaluate.py.

Numbers (PSNR/SSIM/SAM/ERGAS) tell you IF the model is good; this tells
you and your judges WHERE and HOW -- side-by-side LR input / naive
bicubic upsample / model output / real ground truth, so a viewer can
see the model is doing more than naive interpolation, not just trust
a number on a slide.

Usage:
    python -m src.eval.visualize
    python -m src.eval.visualize --num_samples 8 --split test
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.transform import resize

from src.models.rrdb import RRDBNet
from src.data.dataset import SatelliteSRDataset
from src.eval.metrics import compute_psnr, compute_ssim


def to_uint8_rgb(img_chw: np.ndarray) -> np.ndarray:
    """
    img_chw: (C, H, W) float array, C>=3, bands ordered R,G,B,NIR per
    Module 2's evalscript. Takes first 3 channels, does a simple
    percentile stretch for viewing -- raw reflectance values are too
    dim/flat to look at directly, same approach used in the earlier
    patch_sample.png sanity check.
    """
    rgb = img_chw[:3].transpose(1, 2, 0)   # (H, W, 3)
    rgb = np.clip(rgb, 0, None)
    p99 = np.percentile(rgb, 99) + 1e-8
    rgb = np.clip(rgb / p99, 0, 1)
    return rgb


def bicubic_baseline(lr_chw: np.ndarray, target_hw) -> np.ndarray:
    """
    Naive upsampling baseline for visual/numeric comparison -- makes it
    obvious in the figure (and in the printed metrics) that the model is
    doing more than what free interpolation already gives you.
    """
    lr_hwc = lr_chw.transpose(1, 2, 0)
    up = resize(lr_hwc, (*target_hw, lr_hwc.shape[-1]), order=3,
                mode="reflect", anti_aliasing=True)
    return up.transpose(2, 0, 1).astype(np.float32)


def visualize(checkpoint_path="checkpoints/best_model_phase2.pt", split="test",
              num_samples=6, out_path="demo/visual_eval_phase2.png", seed=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(checkpoint_path, map_location=device)
    num_blocks = ckpt.get("num_blocks", 16)   # match evaluate.py's fix -- read
                                                # the actual trained architecture,
                                                # don't hardcode a guess
    model = RRDBNet(in_channels=4, out_channels=4, num_blocks=num_blocks, scale_factor=4).to(device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()

    dataset = SatelliteSRDataset(augment=False)   # no augmentation for evaluation,
                                                     # same reasoning as evaluate.py
    with open(f"data/splits/{split}.json") as f:
        files = set(json.load(f))
    indices = [i for i, f in enumerate(dataset.patch_files) if str(f) in files]

    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(indices, size=min(num_samples, len(indices)), replace=False)

    fig, axes = plt.subplots(len(sample_indices), 4, figsize=(14, 3.2 * len(sample_indices)))
    if len(sample_indices) == 1:
        axes = axes[None, :]   # keep 2D indexing consistent for a single sample

    col_titles = ["LR input (bicubic-resized for display)", "Bicubic baseline", "Model SR output", "Ground truth HR"]

    with torch.no_grad():
        for row, idx in enumerate(sample_indices):
            lr_t, hr_t = dataset[idx]
            lr_np = lr_t.numpy()
            hr_np = hr_t.numpy()

            pred_t = model(lr_t.unsqueeze(0).to(device))[0].cpu().clamp(0, 1)
            pred_np = pred_t.numpy()

            bicubic_np = np.clip(bicubic_baseline(lr_np, hr_np.shape[1:]), 0, 1)

            psnr_model = compute_psnr(pred_np.transpose(1, 2, 0), hr_np.transpose(1, 2, 0))
            ssim_model = compute_ssim(pred_np.transpose(1, 2, 0), hr_np.transpose(1, 2, 0))
            psnr_bicubic = compute_psnr(bicubic_np.transpose(1, 2, 0), hr_np.transpose(1, 2, 0))
            ssim_bicubic = compute_ssim(bicubic_np.transpose(1, 2, 0), hr_np.transpose(1, 2, 0))

            patch_name = dataset.patch_files[idx].name
            lr_display = bicubic_baseline(lr_np, hr_np.shape[1:])   # just for viewing at HR size

            panels = [to_uint8_rgb(lr_display), to_uint8_rgb(bicubic_np),
                      to_uint8_rgb(pred_np), to_uint8_rgb(hr_np)]
            panel_captions = [
                None,
                f"PSNR {psnr_bicubic:.1f} / SSIM {ssim_bicubic:.3f}",
                f"PSNR {psnr_model:.1f} / SSIM {ssim_model:.3f}",
                None,
            ]

            for col, (panel, caption) in enumerate(zip(panels, panel_captions)):
                ax = axes[row, col]
                ax.imshow(panel)
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(col_titles[col], fontsize=9)
                if caption:
                    ax.set_xlabel(caption, fontsize=8)
            axes[row, 0].set_ylabel(patch_name, fontsize=7)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved visual comparison grid: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model_phase2.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num_samples", type=int, default=6)
    parser.add_argument("--out", default="demo/visual_eval_phase2.png")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    visualize(checkpoint_path=args.checkpoint, split=args.split,
              num_samples=args.num_samples, out_path=args.out, seed=args.seed)