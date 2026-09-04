import torch
from torch.utils.data import DataLoader
import json
from pathlib import Path

from src.data.dataset import SatelliteSRDataset
from src.models.rrdb import RRDBNet
from src.losses.losses import CombinedSRLoss

def get_dataloaders(batch_size=4):
    patches_dir = "data/patches"
    if not list(Path(patches_dir).glob("*.npy")):
        from src.data.patchify import build_patch_dataset
        build_patch_dataset()

    splits_dir = "data/splits"
    if not Path(f"{splits_dir}/train.json").exists():
        from src.data.splits import create_splits
        create_splits()

    full_dataset = SatelliteSRDataset(patches_dir=patches_dir)

    with open(f"{splits_dir}/train.json") as f:
        train_files = set(json.load(f))
    with open(f"{splits_dir}/val.json") as f:
        val_files = set(json.load(f))

    train_indices = [i for i, f in enumerate(full_dataset.patch_files) if str(f) in train_files]
    val_indices = [i for i, f in enumerate(full_dataset.patch_files) if str(f) in val_files]

    if not train_indices:
        train_indices = list(range(max(1, len(full_dataset) - 2)))
        val_indices = list(range(max(1, len(full_dataset) - 2), len(full_dataset)))

    train_ds = torch.utils.data.Subset(full_dataset, train_indices)
    val_ds = torch.utils.data.Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader

def train(num_epochs=10, lr=2e-4, batch_size=4, accumulation_steps=4, checkpoint_dir="checkpoints"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==========================================")
    print(f" Starting SRM RRDBNet Training on: {device}")
    print(f"==========================================")

    train_loader, val_loader = get_dataloaders(batch_size=batch_size)
    model = RRDBNet(in_channels=4, out_channels=4, num_blocks=8, scale_factor=4).to(device)
    criterion = CombinedSRLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_loss = float("inf")
    Path(checkpoint_dir).mkdir(exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, (lr_batch, hr_batch) in enumerate(train_loader):
            lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(lr_batch)
                loss = criterion(pred, hr_batch) / accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * accumulation_steps

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lr_batch, hr_batch in val_loader:
                lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = model(lr_batch)
                    val_loss += criterion(pred, hr_batch).item()

        val_loss /= max(1, len(val_loader))
        scheduler.step()

        print(f"Epoch {epoch+1:02d}/{num_epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, f"{checkpoint_dir}/best_model.pt")
            print(f"  → Saved new best checkpoint (val_loss: {val_loss:.4f})")

        torch.save(model.state_dict(), f"{checkpoint_dir}/latest_model.pt")

    print("\nTraining complete! Best checkpoint saved to checkpoints/best_model.pt")

if __name__ == "__main__":
    train()
