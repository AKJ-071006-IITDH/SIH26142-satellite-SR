import argparse
import os
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.transform import resize
from tqdm import tqdm

from src.models.rrdb import RRDBNet
from src.models.rrdbnet_attn import AttentionRRDBNet
from src.data.dataset import SatelliteSRDataset

def load_model(checkpoint_path, device):
    """
    Loads either architecture from a checkpoint, auto-detected from the
    "arch" key train_v2.py/train_v2_fidelity.py save (arch="attn_rrdb").
    Older RRDBNet checkpoints (train.py, train_phase4.py, ...) don't have
    that key and fall back to the original architecture, same as before.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    num_blocks = ckpt.get("num_blocks", 16)
    if ckpt.get("arch") == "attn_rrdb":
        model = AttentionRRDBNet(in_channels=4, out_channels=4, num_blocks=num_blocks, scale_factor=4).to(device)
    else:
        model = RRDBNet(in_channels=4, out_channels=4, num_blocks=num_blocks, scale_factor=4).to(device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()
    return model

def to_uint8_rgb(img_chw: np.ndarray) -> np.ndarray:
    """Takes first 3 channels (RGB) and applies percentile stretch for viewing."""
    rgb = img_chw[:3].transpose(1, 2, 0)   # (H, W, 3)
    rgb = np.clip(rgb, 0, None)
    p99 = np.percentile(rgb, 99) + 1e-8
    rgb = np.clip(rgb / p99, 0, 1)
    return rgb

def resize_for_display(lr_chw: np.ndarray, target_hw) -> np.ndarray:
    """Resizes the LR image so it matches the HR dimensions for side-by-side plotting."""
    lr_hwc = lr_chw.transpose(1, 2, 0)
    up = resize(lr_hwc, (*target_hw, lr_hwc.shape[-1]), order=0, # Nearest neighbor interpolation
                mode="reflect", anti_aliasing=False)
    return up.transpose(2, 0, 1).astype(np.float32)

def visualize_all(checkpoint_path="checkpoints/best_model_phase2.pt", split="test", out_dir="demo/full_test_set"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Setup Model
    model = load_model(checkpoint_path, device)

    # 2. Setup Dataset
    dataset = SatelliteSRDataset(augment=False)
    with open(f"data/splits/{split}.json") as f:
        files = set(json.load(f))
    indices = [i for i, f in enumerate(dataset.patch_files) if str(f) in files]

    # 3. Create Output Directory
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print(f"Saving {len(indices)} test patches to {out_dir}/...")

    col_titles = ["LR Input", "Model SR Output", "Ground Truth HR"]
    
    with torch.no_grad():
        for idx in tqdm(indices, desc="Generating Visuals"):
            lr_t, hr_t = dataset[idx]
            lr_np = lr_t.numpy()
            hr_np = hr_t.numpy()
            target_hw = (hr_t.shape[1], hr_t.shape[2])
            patch_name = dataset.patch_files[idx].stem # Gets name without .npy

            # Get Model Prediction
            pred_t = model(lr_t.unsqueeze(0).to(device))[0].cpu().clamp(0, 1)
            pred_np = pred_t.numpy()

            # Resize LR just for the plot
            lr_display = resize_for_display(lr_np, target_hw)

            # Plotting the 3 columns
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            panels = [to_uint8_rgb(lr_display), to_uint8_rgb(pred_np), to_uint8_rgb(hr_np)]

            for col, panel in enumerate(panels):
                ax = axes[col]
                ax.imshow(panel)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(col_titles[col], fontsize=12)
            
            fig.suptitle(patch_name, fontsize=10)
            plt.tight_layout()
            
            # Save and close the figure
            out_path = os.path.join(out_dir, f"{patch_name}.png")
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model_phase2.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out_dir", default="demo/full_test_set")
    args = parser.parse_args()
    visualize_all(checkpoint_path=args.checkpoint, split=args.split, out_dir=args.out_dir)