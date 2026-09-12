# src/eval/perceptual_metrics.py
"""
Perceptual, reference-based metrics -- for judging whether an output *looks*
like the ground truth, which PSNR/SSIM/SAM/ERGAS cannot tell you.

Those four are distortion metrics: they ask whether each pixel holds the right
number. A blurry prediction that hedges toward the local average scores well on
them precisely because hedging minimises squared error. That is why the attn
models can sit at 37+ dB and still look smoother than the ground truth.

Two metrics here, both compared against the real HR target:

  LPIPS            deep-feature distance, the field standard for this
                   situation. Lower is better. Tracks human judgement far
                   better than PSNR. Uses AlexNet features by DEFAULT and on
                   purpose: the GAN training script optimises a VGG19
                   perceptual loss, so judging with VGG here would be circular.

  HF energy ratio  high-pass energy of the prediction divided by high-pass
                   energy of the target. 1.0 = exactly as sharp as the truth,
                   below 1.0 = blurrier, above 1.0 = over-sharpened. No
                   dependencies, and it quantifies the specific failure
                   (missing texture) in a number that explains itself.

Usage:
    python -m src.eval.perceptual_metrics --checkpoint checkpoints/best_model_attn.pt
    python -m src.eval.perceptual_metrics --compare      # all known checkpoints
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.data.dataset import SatelliteSRDataset
from src.models.rrdb import RRDBNet
from src.models.rrdbnet_attn import AttentionRRDBNet
from src.eval.metrics import evaluate_batch


def hf_energy_ratio(pred, target, eps=1e-8):
    """
    Ratio of high-frequency energy in pred vs target.

    High-pass is (image - blurred image) using a 3x3 box blur; energy is the
    mean square of that residual. Computed per band and averaged, so it is not
    dominated by whichever band happens to be brightest.

    1.0 means the prediction carries the same amount of fine detail as the
    ground truth. Blur shows up as a value well below 1.0 -- which is exactly
    what a pixel-loss-trained SR model produces, and exactly what PSNR hides.
    """
    C = pred.shape[1]
    k = torch.ones(C, 1, 3, 3, device=pred.device, dtype=pred.dtype) / 9.0
    hp_p = pred - F.conv2d(pred, k, padding=1, groups=C)
    hp_t = target - F.conv2d(target, k, padding=1, groups=C)
    e_p = hp_p.pow(2).mean(dim=(2, 3))
    e_t = hp_t.pow(2).mean(dim=(2, 3))
    return (e_p / (e_t + eps)).mean().item()


def _to_lpips_rgb(x):
    """4-band reflectance -> 3-channel RGB in [-1, 1], which is what LPIPS expects."""
    rgb = x[:, :3].clamp(0, 1)
    return rgb * 2.0 - 1.0


class LPIPSScorer:
    """Thin wrapper so a missing `lpips` package degrades to a clear message, not a crash."""

    def __init__(self, net="alex", device="cpu"):
        self.device = device
        try:
            import lpips
        except ImportError:
            self.fn = None
            self.error = ("lpips not installed -- run `pip install lpips` to enable it. "
                          "HF energy ratio is still reported below.")
            return
        self.error = None
        self.fn = lpips.LPIPS(net=net, verbose=False).to(device)
        for p in self.fn.parameters():
            p.requires_grad = False

    @property
    def available(self):
        return self.fn is not None

    def __call__(self, pred, target):
        # move inputs to the scorer's own device -- callers legitimately hold
        # CPU tensors (compare_models works with numpy panels) and should not
        # each have to remember where the LPIPS network happens to live
        with torch.no_grad():
            return self.fn(_to_lpips_rgb(pred.to(self.device)),
                            _to_lpips_rgb(target.to(self.device))).mean().item()


def load_model(checkpoint_path, device):
    """Same architecture auto-detection as src/eval/visualize.py."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    num_blocks = ckpt.get("num_blocks", 16)
    if ckpt.get("arch") == "attn_rrdb":
        model = AttentionRRDBNet(in_channels=4, out_channels=4, num_blocks=num_blocks,
                                  scale_factor=4).to(device)
    else:
        model = RRDBNet(in_channels=4, out_channels=4, num_blocks=num_blocks,
                         scale_factor=4).to(device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()
    return model


def build_loader(split="test", limit=None, batch_size=8, seed=0, manifest=None):
    """
    manifest=None uses the original patch-level split (data/splits/*.json).
    Passing a manifest path uses the leak-free spatial/region split instead --
    necessary for any model trained by train_v3.py, and the honest yardstick
    for the others too, since the old split shares tiles across train and test.
    """
    if manifest:
        from src.data.tile_dataset import TileSRDataset
        dataset = TileSRDataset(split, manifest_path=manifest, augment=False)
        indices = list(range(len(dataset)))
        if limit is not None and limit < len(indices):
            rng = np.random.default_rng(seed)
            indices = sorted(rng.choice(indices, size=limit, replace=False).tolist())
            dataset = Subset(dataset, indices)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                           num_workers=0), len(indices)

    dataset = SatelliteSRDataset(augment=False)
    with open(f"data/splits/{split}.json") as f:
        files = set(json.load(f))
    indices = [i for i, f in enumerate(dataset.patch_files) if str(f) in files]
    if limit is not None and limit < len(indices):
        rng = np.random.default_rng(seed)
        indices = sorted(rng.choice(indices, size=limit, replace=False).tolist())
    return DataLoader(Subset(dataset, indices), batch_size=batch_size,
                       shuffle=False, num_workers=0), len(indices)


def score(model, loader, lpips_scorer, device, include_bicubic=False):
    acc = {"psnr": [], "ssim": [], "sam": [], "ergas": [], "hf": [], "lpips": []}
    with torch.no_grad():
        for lr_b, hr_b in loader:
            lr_b, hr_b = lr_b.to(device), hr_b.to(device)
            pred = (F.interpolate(lr_b, scale_factor=4, mode="bicubic", align_corners=False)
                    if include_bicubic else model(lr_b))
            for k, v in evaluate_batch(pred, hr_b, scale_factor=4).items():
                acc[k].append(v)
            acc["hf"].append(hf_energy_ratio(pred.clamp(0, 1), hr_b))
            if lpips_scorer.available:
                acc["lpips"].append(lpips_scorer(pred, hr_b))
    return {k: (float(np.mean(v)) if v else float("nan")) for k, v in acc.items()}


DEFAULT_TARGETS = [
    ("bicubic",       None),
    ("rrdb_phase2",   "checkpoints/best_model_phase2.pt"),
    ("attn",          "checkpoints/best_model_attn.pt"),
    ("attn_fidelity", "checkpoints/best_model_attn_fidelity.pt"),
    ("attn_gan",      "checkpoints/best_model_attn_gan.pt"),
    ("attn_gan_v2",   "checkpoints/best_model_attn_gan_v2.pt"),
    ("attn_gan_v3",   "checkpoints/best_model_attn_gan_v3.pt"),
    ("attn_v3",       "checkpoints/best_model_attn_v3.pt"),
    ("attn_icnr",     "checkpoints/best_model_attn_icnr.pt"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--compare", action="store_true",
                         help="score every known checkpoint that exists on disk")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None,
                         help="score a random subset of this many patches (default: all)")
    parser.add_argument("--lpips_net", default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--manifest", nargs="?", const="data/splits/manifest.json",
                         default=None,
                         help="score on the leak-free manifest split instead of "
                              "data/splits/*.json (required for train_v3 models)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader, n = build_loader(args.split, args.limit, manifest=args.manifest)
    source = f"manifest ({args.manifest})" if args.manifest else "data/splits/*.json"
    print(f"Scoring on {n} crops from the '{args.split}' split · source: {source} "
          f"(device: {device})\n")

    scorer = LPIPSScorer(net=args.lpips_net, device=device)
    if not scorer.available:
        print(f"NOTE: {scorer.error}\n")
    else:
        print(f"LPIPS backbone: {args.lpips_net} "
              f"(deliberately not VGG -- the GAN script trains against VGG19)\n")

    targets = DEFAULT_TARGETS if args.compare else [("model", args.checkpoint)]
    rows = []
    for name, path in targets:
        is_bicubic = path is None
        if not is_bicubic:
            if path is None or not Path(path).exists():
                continue
            model = load_model(path, device)
        else:
            model = None
        rows.append((name, score(model, loader, scorer, device, include_bicubic=is_bicubic)))

    hdr = f"{'model':<16}{'PSNR':>9}{'SSIM':>9}{'SAM':>9}{'ERGAS':>9}{'LPIPS':>10}{'HF ratio':>11}"
    print(hdr)
    print("-" * len(hdr))
    for name, m in rows:
        print(f"{name:<16}{m['psnr']:>9.3f}{m['ssim']:>9.4f}{m['sam']:>9.3f}"
              f"{m['ergas']:>9.3f}{m['lpips']:>10.4f}{m['hf']:>11.3f}")
    print("-" * len(hdr))
    print("PSNR/SSIM up = better · SAM/ERGAS/LPIPS down = better · HF ratio: 1.000 = as sharp as truth")


if __name__ == "__main__":
    main()
