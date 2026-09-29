# src/eval/benchmark_speed.py
"""
Inference speed per 100 km^2 of output.

Output is 10 m/pixel, so 100 km^2 = 10 km x 10 km = 1000 x 1000 HR pixels,
produced from a 250 x 250 LR input at x4.

Uses a real 1000 x 1000 region of a raw tile (degraded to LR exactly as in
training) so the input statistics are realistic.

Two inference modes, because they cost differently:
  full   -- the whole 250x250 LR frame in one forward pass. The model is fully
            convolutional, so this works if it fits in VRAM.
  tiled  -- the frame cut into 32x32 LR tiles (the training size), batched.
            What you'd use for scenes too large for one pass.

GPU timing notes: CUDA launches are asynchronous, so the clock is only read
after torch.cuda.synchronize(); the first calls include allocation and kernel
selection, so they are discarded as warmup. Reported time is the median of the
timed repeats.

MC dropout (the web app's uncertainty map) runs the model N times, so its cost
is reported separately -- it multiplies inference time by roughly N.

Usage:
    python -m src.eval.benchmark_speed
    python -m src.eval.benchmark_speed --checkpoint checkpoints/best_model_attn_gan_v3.pt --mc_samples 20
"""

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from src.data.degrade import degrade
from src.eval.visualize import load_model
from src.uncertainity.mc_dropout import enable_dropout

HR_SIDE = 1000          # 1000 px x 10 m = 10 km  ->  100 km^2
SCALE = 4
LR_SIDE = HR_SIDE // SCALE
TILE = 32               # LR tile size used in training


def load_region(raw_tiles_dir="data/raw_tiles"):
    """A real 1000x1000 HR region from the first raw tile, degraded to LR."""
    tile_path = sorted(Path(raw_tiles_dir).glob("*.npy"))[0]
    tile = np.load(tile_path, mmap_mode="r")
    hr = np.array(tile[:HR_SIDE, :HR_SIDE], dtype=np.float32)
    lr = degrade(hr, scale_factor=SCALE)                      # (250, 250, 4)
    print(f"Input: {HR_SIDE}x{HR_SIDE} HR region of {tile_path.stem} "
          f"-> {lr.shape[0]}x{lr.shape[1]} LR  (= 100 km^2 at 10 m)")
    return torch.from_numpy(lr.transpose(2, 0, 1).copy()).unsqueeze(0)


def tile_batch(lr):
    """Cut the LR frame into non-overlapping 32x32 tiles (padded to a multiple)."""
    _, c, h, w = lr.shape
    ph, pw = (-h) % TILE, (-w) % TILE
    lr = torch.nn.functional.pad(lr, (0, pw, 0, ph), mode="reflect")
    tiles = lr.unfold(2, TILE, TILE).unfold(3, TILE, TILE)   # 1,C,nh,nw,T,T
    return tiles.permute(0, 2, 3, 1, 4, 5).reshape(-1, c, TILE, TILE)


def time_fn(fn, device, warmup, repeats):
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best_model_attn_gan_v3.pt")
    p.add_argument("--batch", type=int, default=16, help="tiles per batch in tiled mode")
    p.add_argument("--mc_samples", type=int, default=20, help="MC-dropout passes to time")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--fp32", action="store_true", help="disable bf16/fp16 autocast")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.fp32
    amp_dtype = (torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported()
                 else torch.float16)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} · "
              f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB · "
              f"precision: {amp_dtype if use_amp else 'fp32'}")
    else:
        print("WARNING: no CUDA device -- these are CPU timings, not GPU timings")

    model = load_model(args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.checkpoint} ({n_params:,} params)\n")

    lr = load_region().to(device)
    tiles = tile_batch(lr)

    def forward(x):
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            return model(x)

    def run_full():
        forward(lr)

    def run_tiled():
        for i in range(0, tiles.shape[0], args.batch):
            forward(tiles[i:i + args.batch])

    results = []
    for name, fn in (("full frame, 1 pass", run_full),
                     (f"tiled 32x32, batch {args.batch}", run_tiled)):
        try:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t = time_fn(fn, device, args.warmup, args.repeats)
            mem = (torch.cuda.max_memory_allocated() / 2**20
                   if device.type == "cuda" else float("nan"))
            results.append((name, t, mem))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            results.append((name, None, None))

    # MC dropout: N stochastic passes, dropout active, everything else in eval
    model.eval()
    enable_dropout(model)

    def run_mc():
        for _ in range(args.mc_samples):
            run_tiled()

    try:
        t_mc = time_fn(run_mc, device, 1, max(3, args.repeats // 3))
    except torch.cuda.OutOfMemoryError:
        t_mc = None
    model.eval()

    print(f"{'mode':<30}{'time / 100 km^2':>18}{'km^2 per sec':>16}{'peak VRAM':>13}")
    print("-" * 77)
    for name, t, mem in results:
        if t is None:
            print(f"{name:<30}{'out of memory':>18}")
            continue
        print(f"{name:<30}{t * 1000:>15.1f} ms{100 / t:>16.1f}{mem:>10.0f} MB")
    if t_mc is not None:
        print(f"{'MC dropout x' + str(args.mc_samples) + ' (tiled)':<30}"
              f"{t_mc * 1000:>15.1f} ms{100 / t_mc:>16.1f}")
    print("-" * 77)

    best = min((t for _, t, _ in results if t), default=None)
    if best:
        print(f"\nSpeed: {best * 1000:.0f} ms per 100 km^2 on "
              f"{torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}"
              f" ({100 / best:.0f} km^2/s)")
        # context: a full Sentinel-2 granule is 110 km x 110 km
        print(f"       ~{best * 121:.1f} s for a full 12,100 km^2 Sentinel-2 granule")
        if t_mc:
            print(f"       with uncertainty map (MC x{args.mc_samples}): "
                  f"{t_mc:.2f} s per 100 km^2")


if __name__ == "__main__":
    main()
