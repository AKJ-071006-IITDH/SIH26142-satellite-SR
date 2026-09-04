# src/eval/plot_history.py
"""
Plots train vs val loss across epochs from a training_history_*.json file
(the {"train_loss": [...], "val_loss": [...]} dict saved by train.py,
train_phase4.py, and train_v2.py). Architecture-agnostic -- just reads the
history dict, no model imports -- so it works for any of the training
scripts' history files.

Usage:
    python -m src.eval.plot_history --history checkpoints/training_history_attn.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_loss_curve(history: dict, save_path: str, title: str = "Training vs Validation Loss"):
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    epochs = range(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label="train_loss", marker="o", markersize=3)
    ax.plot(epochs, val_loss, label="val_loss", marker="o", markersize=3)

    # Mark the best (lowest) val_loss epoch -- the one the checkpoint saver
    # actually kept -- so it's obvious at a glance where training peaked
    # vs. where it kept running (e.g. into early-stopping patience).
    best_epoch = min(range(len(val_loss)), key=lambda i: val_loss[i])
    ax.axvline(best_epoch + 1, color="gray", linestyle="--", linewidth=1,
               label=f"best val_loss (epoch {best_epoch + 1})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved loss curve to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", default="checkpoints/training_history_attn.json")
    parser.add_argument("--out", default=None,
                         help="Defaults to the history filename with .png instead of .json")
    parser.add_argument("--title", default="Training vs Validation Loss")
    args = parser.parse_args()

    with open(args.history) as f:
        history = json.load(f)

    out_path = args.out or str(Path(args.history).with_suffix(".png"))
    plot_loss_curve(history, out_path, args.title)
