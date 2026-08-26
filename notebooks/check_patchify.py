import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

patch_dir = Path("data/patches")
files = sorted(patch_dir.glob("*.npy"))
sample = files[::len(files)//9][:9]   # 9 evenly-spaced patches

fig, axes = plt.subplots(3, 3, figsize=(9, 9))
for ax, f in zip(axes.flat, sample):
    p = np.load(f)                     # (128,128,4): R,G,B,NIR
    rgb = p[..., :3]
    rgb = np.clip(rgb / np.percentile(rgb, 99), 0, 1)   # simple stretch for viewing
    ax.imshow(rgb)
    ax.set_title(f.name, fontsize=7)
    ax.axis("off")
plt.tight_layout()
plt.savefig("patch_sample.png", dpi=150)
print("saved patch_sample.png")


import numpy as np
from pathlib import Path
bad = [f.name for f in Path("data/patches").glob("*.npy") if np.isnan(np.load(f)).any()]
print(f"{len(bad)} patches with NaNs" if bad else "no NaNs found")