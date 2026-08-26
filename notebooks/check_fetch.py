"""
Sanity check for Module 2 (data fetching).

Run this AFTER `python -m src.data.fetch` completes, before moving on to
patchify (Module 3). This is deliberately a standalone script, not part
of src/ -- it's exploratory/visual verification, not reusable pipeline
code (see the project structure note: notebooks/ is for scratch work,
src/ is for anything the pipeline depends on).

Usage:
    python notebooks/check_fetch.py
    python notebooks/check_fetch.py --tile goa_coastal_winter
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def check_tile(tile_path: Path):
    if not tile_path.exists():
        print(f"File not found: {tile_path}")
        print("Did you run `python -m src.data.fetch` first?")
        sys.exit(1)

    tile = np.load(tile_path)
    print(f"Checking: {tile_path.name}")
    print(f"  Shape: {tile.shape}  (expect roughly (~1100, ~1100, 4))")
    print(f"  Value range: [{tile.min():.4f}, {tile.max():.4f}]")

    nonzero_frac = np.mean(tile != 0)
    print(f"  Non-zero fraction: {nonzero_frac:.1%}")
    if nonzero_frac < 0.5:
        print("  WARNING: more than half the tile is zero -- likely heavy "
              "cloud cover or a bad bbox/date range for this region.")

    # RGB is bands 0,1,2 (Red, Green, Blue) per the evalscript in fetch.py
    rgb = tile[:, :, :3]
    # Min-max stretch for display -- raw reflectance values are small
    # floats and look near-black without this
    rgb_display = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

    nir = tile[:, :, 3]
    nir_display = (nir - nir.min()) / (nir.max() - nir.min() + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(rgb_display)
    axes[0].set_title(f"{tile_path.stem} — RGB")
    axes[0].axis("off")

    axes[1].imshow(nir_display, cmap="Greys_r")
    axes[1].set_title(f"{tile_path.stem} — NIR band")
    axes[1].axis("off")

    plt.tight_layout()
    out_path = Path("notebooks") / f"check_{tile_path.stem}.png"
    plt.savefig(out_path, dpi=100)
    print(f"  Saved preview image to: {out_path}")
    plt.show()

    print("\nWhat to look for:")
    print("  - RGB panel should look like a real place (fields, coastline,")
    print("    buildings, etc.), NOT random noise or solid black/white.")
    print("  - NIR panel: vegetation should appear BRIGHT (NIR reflects")
    print("    strongly off healthy plant matter), water/urban areas darker.")
    print("  - If either looks wrong, check your bbox coordinates and")
    print("    time_interval in configs/config.yaml before trusting this data.")


def check_all_tiles(raw_tiles_dir="data/raw_tiles"):
    tiles = sorted(Path(raw_tiles_dir).glob("*.npy"))
    if not tiles:
        print(f"No tiles found in {raw_tiles_dir}/ -- run `python -m src.data.fetch` first.")
        sys.exit(1)
    print(f"Found {len(tiles)} tile(s). Checking the first one in detail; "
          f"pass --tile <name> to check a specific one.\n")
    check_tile(tiles[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile", type=str, default=None,
                         help="Tile name without extension, e.g. 'chennai_urban_post_monsoon'")
    parser.add_argument("--raw_tiles_dir", type=str, default="data/raw_tiles")
    args = parser.parse_args()

    if args.tile:
        check_tile(Path(args.raw_tiles_dir) / f"{args.tile}.npy")
    else:
        check_all_tiles(args.raw_tiles_dir)