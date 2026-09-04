# src/eval/evaluate_v2.py — eval entry point for the attention-RRDB track (src/models/rrdbnet_attn.py).
import argparse
import torch, json
import numpy as np
from src.models.rrdbnet_attn import AttentionRRDBNet
from src.data.dataset import SatelliteSRDataset
from src.eval.metrics import evaluate_batch
from torch.utils.data import DataLoader, Subset


def run_evaluation(checkpoint_path="checkpoints/best_model_attn.pt", split="test"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(checkpoint_path, map_location=device)
    num_blocks = ckpt.get("num_blocks", 8)
    model = AttentionRRDBNet(in_channels=4, out_channels=4, num_blocks=num_blocks, scale_factor=4).to(device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()

    dataset = SatelliteSRDataset(augment=False)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model_attn.pt")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    run_evaluation(args.checkpoint, args.split)
