# src/eval/evaluate.py
import torch, json
import numpy as np
from src.models.rrdb import RRDBNet
from src.data.dataset import SatelliteSRDataset
from src.eval.metrics import evaluate_batch
from torch.utils.data import DataLoader, Subset

def run_evaluation(checkpoint_path="checkpoints/best_model_phase2.pt", split="test"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(checkpoint_path, map_location=device)
    # Use the num_blocks the checkpoint was actually trained with, not a
    # hardcoded guess — train.py's default (6) differs from this file's
    # old hardcoded value (16), which would otherwise crash on load with
    # a state_dict size mismatch.
    num_blocks = ckpt.get("num_blocks", 16)
    model = RRDBNet(in_channels=4, out_channels=4, num_blocks=num_blocks, scale_factor=4).to(device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()   # critical — see Module 6's note on train() vs eval()

    dataset = SatelliteSRDataset(augment=False)  # NO augmentation for evaluation —
                                                   # you want to measure real
                                                   # performance, not performance
                                                   # on randomly flipped/rotated data
    with open(f"data/splits/{split}.json") as f:
        files = set(json.load(f))
    indices = [i for i, f in enumerate(dataset.patch_files) if str(f) in files]
    loader = DataLoader(Subset(dataset, indices), batch_size=8, shuffle=False)

    all_metrics = {"psnr": [], "ssim": [], "sam": [], "ergas": []}
    with torch.no_grad():
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            pred = model(lr)
            batch_metrics = evaluate_batch(pred, hr, scale_factor=4)
            for k, v in batch_metrics.items():
                all_metrics[k].append(v)

    final = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    print(f"Test set results ({len(indices)} patches):")
    for k, v in final.items():
        print(f"  {k.upper()}: {v:.4f}")
    return final

if __name__ == "__main__":
    run_evaluation()