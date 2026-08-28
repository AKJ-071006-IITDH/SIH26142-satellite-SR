import torch
import json
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Subset

from src.models.rrdb import RRDBNet
from src.data.dataset import SatelliteSRDataset
from src.eval.metrics import evaluate_batch

def run_evaluation(checkpoint_path="checkpoints/best_model.pt", split="test", scale_factor=4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running evaluation on {device}...")

    if not Path(checkpoint_path).exists():
        print(f"Checkpoint {checkpoint_path} not found. Running untrained model evaluation for baseline...")
        model = RRDBNet(in_channels=4, out_channels=4, num_blocks=8, scale_factor=scale_factor).to(device)
    else:
        model = RRDBNet(in_channels=4, out_channels=4, num_blocks=8, scale_factor=scale_factor).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)

    model.eval()
    dataset = SatelliteSRDataset(augment=False, scale_factor=scale_factor)

    split_file = Path(f"data/splits/{split}.json")
    if not split_file.exists():
        from src.data.splits import create_splits
        create_splits()

    with open(split_file) as f:
        files = set(json.load(f))

    indices = [i for i, f in enumerate(dataset.patch_files) if str(f) in files]
    if not indices:
        indices = list(range(min(len(dataset), 10)))

    loader = DataLoader(Subset(dataset, indices), batch_size=4, shuffle=False)

    all_metrics = {"psnr": [], "ssim": [], "sam": [], "ergas": []}
    with torch.no_grad():
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            pred = model(lr)
            batch_m = evaluate_batch(pred, hr, scale_factor=scale_factor)
            for k, v in batch_m.items():
                all_metrics[k].append(v)

    final = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    print(f"\n=== Evaluation Results ({split} set, {len(indices)} patches) ===")
    print(f"  PSNR:  {final['psnr']:.2f} dB")
    print(f"  SSIM:  {final['ssim']:.4f}")
    print(f"  SAM:   {final['sam']:.2f}°")
    print(f"  ERGAS: {final['ergas']:.2f}")
    return final

if __name__ == "__main__":
    run_evaluation()
