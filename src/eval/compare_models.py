# src/eval/compare_models.py
"""
Side-by-side visual comparison of several checkpoints on the same test patches.

visualize.py shows one model at a time, which makes it hard to judge whether a
change actually helped -- you end up flipping between two files. This puts every
model in one row, in the same layout and rendering as visualize.py, so the
differences are directly comparable.

Same conventions as visualize.py by design: it reuses that module's
`to_uint8_rgb` percentile stretch and `bicubic_baseline` resize, so panels look
identical to demo/visual_eval_*.png -- only the number of model columns changes.

Optional --crop magnifies a centre crop. Fine texture is only legible when
magnified, so that mode is for judging whether detail is real structure or just
grain; it is a stress test, not a typical view.

Usage:
    python -m src.eval.compare_models
    python -m src.eval.compare_models --num_samples 6 --out demo/visual_eval_all.png
    python -m src.eval.compare_models --crop 56 --hardest --out demo/visual_eval_zoom.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.data.dataset import SatelliteSRDataset
from src.eval.metrics import compute_psnr, compute_ssim
from src.eval.visualize import load_model, to_uint8_rgb, bicubic_baseline
from src.eval.perceptual_metrics import LPIPSScorer

DEFAULT_CHECKPOINTS = [
    ("rrdb_phase2",   "checkpoints/best_model_phase2.pt"),
    ("attn",          "checkpoints/best_model_attn.pt"),
    ("attn_fidelity", "checkpoints/best_model_attn_fidelity.pt"),
    ("attn_gan",      "checkpoints/best_model_attn_gan.pt"),
    ("attn_gan_v3",   "checkpoints/best_model_attn_gan_v3.pt"),
    ("attn_v3",       "checkpoints/best_model_attn_v3.pt"),
]


def pick_patches(dataset, indices, n, seed, hardest):
    """Random by default; --hardest picks the highest-texture patches (worst case)."""
    if hardest:
        scored = []
        for i in indices[:: max(1, len(indices) // 120)]:
            _, hr = dataset[i]
            hp = hr - F.avg_pool2d(hr.unsqueeze(0), 3, 1, 1)[0]
            scored.append((hp.pow(2).mean().item(), i))
        scored.sort(reverse=True)
        return [i for _, i in scored[:n]]
    rng = np.random.default_rng(seed)
    return rng.choice(indices, size=min(n, len(indices)), replace=False)


def compare(checkpoints=None, split="test", num_samples=4, out_path="demo/visual_eval_all.png",
            seed=0, crop=None, hardest=False, manifest=None):
    """
    manifest=None uses the original patch-level split. Pass a manifest path to
    use the leak-free split instead -- REQUIRED for train_v3 models, whose
    training band overlaps some of data/splits/test.json.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if checkpoints:
        targets = [(Path(p).stem.replace("best_model_", ""), p) for p in checkpoints]
    else:
        targets = [(n, p) for n, p in DEFAULT_CHECKPOINTS if Path(p).exists()]
    if not targets:
        raise SystemExit("No checkpoints found -- train a model first.")

    models = [(name, load_model(path, device)) for name, path in targets]
    print(f"Comparing: {', '.join(n for n, _ in models)}")

    if manifest:
        from src.data.tile_dataset import TileSRDataset
        dataset = TileSRDataset(split, manifest_path=manifest, augment=False)
        indices = list(range(len(dataset)))
        label_of = lambda i: f"{split}[{i}]"
    else:
        dataset = SatelliteSRDataset(augment=False)   # no augmentation for evaluation
        with open(f"data/splits/{split}.json") as f:
            files = set(json.load(f))
        indices = [i for i, f in enumerate(dataset.patch_files) if str(f) in files]
        label_of = lambda i: dataset.patch_files[i].name

    sample_indices = pick_patches(dataset, indices, num_samples, seed, hardest)
    print(f"{len(sample_indices)} patches from '{split}'"
          f"{' (highest-texture)' if hardest else ' (random)'}"
          f"{' · leak-free manifest' if manifest else ''}")

    lpips_scorer = LPIPSScorer(net="alex", device=device)

    col_titles = ["LR input (bicubic-resized for display)", "Bicubic baseline"] \
                 + [f"{n} SR output" for n, _ in models] + ["Ground truth HR"]
    n_cols = len(col_titles)

    fig, axes = plt.subplots(len(sample_indices), n_cols,
                              figsize=(3.5 * n_cols, 3.2 * len(sample_indices)))
    if len(sample_indices) == 1:
        axes = axes[None, :]

    def crop_of(chw):
        if crop is None:
            return chw
        c = (chw.shape[1] - crop) // 2
        return chw[:, c:c + crop, c:c + crop]

    with torch.no_grad():
        for row, idx in enumerate(sample_indices):
            lr_t, hr_t = dataset[idx]
            lr_np, hr_np = lr_t.numpy(), hr_t.numpy()
            hr_hwc = hr_np.transpose(1, 2, 0)

            bicubic_np = np.clip(bicubic_baseline(lr_np, hr_np.shape[1:]), 0, 1)
            outputs = [("bicubic", bicubic_np)]
            for name, m in models:
                pred = m(lr_t.unsqueeze(0).to(device))[0].cpu().clamp(0, 1).numpy()
                outputs.append((name, pred))

            panels, captions = [bicubic_baseline(lr_np, hr_np.shape[1:])], [None]
            for name, arr in outputs:
                hwc = arr.transpose(1, 2, 0)
                cap = (f"PSNR {compute_psnr(hwc, hr_hwc):.1f} / "
                       f"SSIM {compute_ssim(hwc, hr_hwc):.3f}")
                if lpips_scorer.available:
                    t = torch.from_numpy(arr).unsqueeze(0)
                    cap += f" / LPIPS {lpips_scorer(t, hr_t.unsqueeze(0)):.3f}"
                panels.append(arr)
                captions.append(cap)
            panels.append(hr_np)
            captions.append(None)

            for col, (panel, caption) in enumerate(zip(panels, captions)):
                ax = axes[row, col]
                ax.imshow(to_uint8_rgb(crop_of(panel)),
                          interpolation="nearest" if crop else None)
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(col_titles[col], fontsize=9)
                if caption:
                    ax.set_xlabel(caption, fontsize=8)
            axes[row, 0].set_ylabel(label_of(idx), fontsize=7)

    if crop:
        fig.suptitle(f"{crop}x{crop} centre crops, magnified"
                     f"{' -- highest-texture patches' if hardest else ''}", fontsize=11)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visual comparison grid: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="*", default=None,
                         help="checkpoint paths; default = all known ones that exist")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--out", default="demo/visual_eval_all.png")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--crop", type=int, default=None,
                         help="magnify an NxN centre crop (e.g. 56); omit for full patch")
    parser.add_argument("--hardest", action="store_true",
                         help="use highest-texture patches instead of random ones")
    parser.add_argument("--manifest", nargs="?", const="data/splits/manifest_spatial.json",
                         default=None,
                         help="use the leak-free manifest split (required for train_v3)")
    args = parser.parse_args()
    compare(checkpoints=args.checkpoints, split=args.split, num_samples=args.num_samples,
            out_path=args.out, seed=args.seed, crop=args.crop, hardest=args.hardest,
            manifest=args.manifest)
