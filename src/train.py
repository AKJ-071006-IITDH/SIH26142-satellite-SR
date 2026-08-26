"""
Module 6: Training loop, now with early stopping and explicit
overfitting-gap monitoring -- so the team doesn't have to babysit
loss curves manually, especially important given the modest dataset
size discussed for this hackathon.
"""

import torch
from torch.utils.data import DataLoader
import json
from pathlib import Path

from src.data.dataset import SatelliteSRDataset
from src.models.rrdb import RRDBNet
from src.losses.losses import CombinedSRLoss


def get_dataloaders(batch_size=16):
    full_dataset = SatelliteSRDataset(patches_dir="data/patches")

    with open("data/splits/train.json") as f:
        train_files = set(json.load(f))
    with open("data/splits/val.json") as f:
        val_files = set(json.load(f))

    train_indices = [i for i, f in enumerate(full_dataset.patch_files) if str(f) in train_files]
    val_indices = [i for i, f in enumerate(full_dataset.patch_files) if str(f) in val_files]

    train_ds = torch.utils.data.Subset(full_dataset, train_indices)
    val_ds = torch.utils.data.Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)
    return train_loader, val_loader


class EarlyStopper:
    """
    Two independent stopping conditions, tracked separately because they
    catch different failure modes:

    1. `patience` -- classic early stopping. If val_loss hasn't improved
       in `patience` epochs, training has plateaued -- further epochs are
       very unlikely to help and are just wasting your limited compute
       time budget.

    2. `overfit_gap_threshold` -- a DIFFERENT signal from plain patience.
       This catches the specific "train_loss keeps dropping while
       val_loss stalls or rises" pattern that's the direct symptom of a
       model with more capacity than a small dataset can constrain
       (exactly the risk flagged for this dataset size). Patience alone
       can miss this if val_loss oscillates without a clean best-then-
       plateau shape -- the gap check catches it directly regardless of
       the exact val_loss trajectory shape.
    """
    def __init__(self, patience=7, overfit_gap_threshold=0.15, min_epochs=10):
        self.patience = patience
        self.overfit_gap_threshold = overfit_gap_threshold
        self.min_epochs = min_epochs
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

    def check(self, epoch, train_loss, val_loss):
        """
        Returns (should_stop: bool, reason: str or None)
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        # Don't apply either stopping rule before min_epochs -- early
        # training is naturally noisy, and stopping too eagerly wastes
        # a run that just needed a few more epochs to get going.
        if epoch < self.min_epochs:
            return False, None

        if self.epochs_without_improvement >= self.patience:
            return True, (f"No val_loss improvement for {self.patience} epochs "
                           f"(best: {self.best_val_loss:.4f})")

        # Relative gap between train and val loss. A small gap is normal
        # and healthy; a LARGE and GROWING gap means the model is
        # memorizing training patches rather than learning general
        # SR behavior -- exactly the risk with a dataset this size.
        if train_loss > 0:
            relative_gap = (val_loss - train_loss) / train_loss
            if relative_gap > self.overfit_gap_threshold:
                return True, (f"Overfitting gap detected: val_loss is "
                               f"{relative_gap:.1%} higher than train_loss "
                               f"(threshold: {self.overfit_gap_threshold:.0%})")

        return False, None


def train(num_epochs=100, lr=2e-4, batch_size=8, checkpoint_dir="checkpoints",
          num_blocks=6, patience=7, overfit_gap_threshold=0.15):
    """
    num_epochs raised to 100 as a ceiling -- early stopping will almost
    certainly halt well before this on a dataset this size, so the ceiling
    just needs to be "high enough to not be the limiting factor", not a
    literal target.

    num_blocks defaulted to 6 (between the earlier 8 and the smaller
    4-5 discussed for the smallest dataset) -- reasonable given the
    larger 9-region x 3-season dataset, adjust down if you still see
    overfitting trigger very early (within the first 15-20 epochs).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    train_loader, val_loader = get_dataloaders(batch_size)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = RRDBNet(in_channels=4, out_channels=4, num_blocks=num_blocks, scale_factor=4).to(device)
    criterion = CombinedSRLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.cuda.amp.GradScaler()   # mixed precision, per the 6GB VRAM tuning

    early_stopper = EarlyStopper(patience=patience,
                                  overfit_gap_threshold=overfit_gap_threshold,
                                  min_epochs=10)

    Path(checkpoint_dir).mkdir(exist_ok=True)
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        # --- Training phase ---
        model.train()
        train_loss = 0.0
        for lr_batch, hr_batch in train_loader:
            lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                pred = model(lr_batch)
                loss = criterion(pred.float(), hr_batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
        train_loss /= len(train_loader)

        # --- Validation phase ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lr_batch, hr_batch in val_loader:
                lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
                with torch.cuda.amp.autocast():
                    pred = model(lr_batch)
                    val_loss += criterion(pred.float(), hr_batch).item()
        val_loss /= len(val_loader)

        scheduler.step()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        gap_pct = ((val_loss - train_loss) / train_loss * 100) if train_loss > 0 else 0
        print(f"Epoch {epoch+1}/{num_epochs} | train: {train_loss:.4f} | "
              f"val: {val_loss:.4f} | gap: {gap_pct:+.1f}% | "
              f"lr: {scheduler.get_last_lr()[0]:.6f}")

        # Save best checkpoint by val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "num_blocks": num_blocks,
            }, f"{checkpoint_dir}/best_model.pt")
            print(f"  -> saved new best checkpoint (val_loss: {val_loss:.4f})")

        torch.save(model.state_dict(), f"{checkpoint_dir}/latest_model.pt")

        # --- Early stopping check ---
        should_stop, reason = early_stopper.check(epoch, train_loss, val_loss)
        if should_stop:
            print(f"\nStopping early at epoch {epoch+1}: {reason}")
            print(f"Best checkpoint retained at checkpoints/best_model.pt "
                  f"(val_loss: {best_val_loss:.4f})")
            break
    else:
        print(f"\nCompleted all {num_epochs} epochs without triggering early stopping.")

    # Save training history for plotting later (e.g. in your presentation)
    with open(f"{checkpoint_dir}/training_history.json", "w") as f:
        json.dump(history, f)

    return history


if __name__ == "__main__":
    train()