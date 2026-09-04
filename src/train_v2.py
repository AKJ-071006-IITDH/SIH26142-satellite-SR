"""
train_v2: training entry point for the attention-enhanced RRDB architecture
(src/models/rrdbnet_attn.py). Trains from scratch -- the state dict isn't
compatible with the original RRDBNet checkpoints -- using a purely
fidelity-driven loss (pixel + SSIM + spectral + self-perceptual, no
adversarial term). Mirrors src/train.py's structure and conventions so the
two tracks stay easy to compare.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
from pathlib import Path
from tqdm import tqdm

from src.data.dataset import SatelliteSRDataset
from src.models.rrdb import RRDBNet
from src.models.rrdbnet_attn import AttentionRRDBNet
from src.losses.losses_attn import CombinedFidelityLossV2
from src.eval.plot_history import plot_loss_curve


def get_dataloaders(batch_size=8):
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
    """Same two-condition early stopping as train.py -- see that file for the rationale."""
    def __init__(self, patience=7, overfit_gap_threshold=0.15, min_epochs=10):
        self.patience = patience
        self.overfit_gap_threshold = overfit_gap_threshold
        self.min_epochs = min_epochs
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

    def check(self, epoch, train_loss, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if epoch < self.min_epochs:
            return False, None

        if self.epochs_without_improvement >= self.patience:
            return True, (f"No val_loss improvement for {self.patience} epochs "
                           f"(best: {self.best_val_loss:.4f})")

        if train_loss > 0:
            relative_gap = (val_loss - train_loss) / train_loss
            if relative_gap > self.overfit_gap_threshold:
                return True, (f"Overfitting gap detected: val_loss is "
                               f"{relative_gap:.1%} higher than train_loss "
                               f"(threshold: {self.overfit_gap_threshold:.0%})")

        return False, None


class FrozenFeatureExtractor(nn.Module):
    """
    Early layers of a SEPARATE, already-trained RRDBNet, used as a fixed
    perceptual feature extractor -- the same trick src/train.py uses.

    This must never be constructed from the model being trained.
    SpectralPerceptualLoss calls requires_grad=False on whatever it is
    handed, and nn.Module attributes are references, not copies -- so
    passing the live model silently freezes its conv_first and first RRDB
    blocks at random initialization.
    """
    def __init__(self, pretrained_model):
        super().__init__()
        self.features = nn.Sequential(
            pretrained_model.conv_first,
            *list(pretrained_model.rrdb_blocks.children())[:3]
        )

    def forward(self, x):
        return self.features(x)


def build_feature_extractor(device, checkpoint_path="checkpoints/best_model.pt"):
    """Loads the trained Phase 1 baseline as the frozen perceptual reference."""
    if not Path(checkpoint_path).exists():
        print(f"WARNING: {checkpoint_path} not found -- training WITHOUT the perceptual "
              f"term. Train the Phase 1 baseline first if you want it.")
        return None

    ckpt = torch.load(checkpoint_path, map_location=device)
    base = RRDBNet(in_channels=4, out_channels=4,
                    num_blocks=ckpt["num_blocks"], scale_factor=4).to(device)
    base.load_state_dict(ckpt["model_state_dict"])
    print(f"Perceptual features from {checkpoint_path} "
          f"(num_blocks={ckpt['num_blocks']}, val_loss={ckpt['val_loss']:.4f})")
    return FrozenFeatureExtractor(base).to(device)


def train(num_epochs=100, lr=1e-4, batch_size=8, checkpoint_dir="checkpoints",
          num_blocks=8, patience=7, overfit_gap_threshold=0.15,
          warmup_steps=300, grad_clip_norm=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training AttentionRRDBNet on: {device}")

    train_loader, val_loader = get_dataloaders(batch_size)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = AttentionRRDBNet(in_channels=4, out_channels=4, num_blocks=num_blocks, scale_factor=4).to(device)

    feature_extractor = build_feature_extractor(device)
    criterion = CombinedFidelityLossV2(feature_extractor=feature_extractor)

    # The loss freezes whatever feature extractor it is given, so verify it
    # didn't reach into the model we're about to train.
    frozen = [name for name, p in model.named_parameters() if not p.requires_grad]
    assert not frozen, (f"{len(frozen)} model parameters are frozen before training "
                        f"(first: {frozen[0]}) -- the loss is holding a reference to "
                        f"the model being trained.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # bfloat16 over float16: bf16 carries fp32's exponent range, so gradients
    # can't overflow the way they do in fp16, where GradScaler multiplies them
    # by ~65536 and anything above 65504 becomes inf. That overflow is normally
    # harmless (the scaler detects it and halves the scale), but repeated hits
    # collapse the scale far enough that small gradients underflow to zero and
    # training degrades. bf16 sidesteps the whole cycle and needs no scaler.
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    needs_scaler = use_amp and amp_dtype is torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=needs_scaler)
    if use_amp:
        print(f"AMP dtype: {amp_dtype} (GradScaler {'on' if needs_scaler else 'off -- not needed for bf16'})")

    early_stopper = EarlyStopper(patience=patience,
                                  overfit_gap_threshold=overfit_gap_threshold,
                                  min_epochs=10)

    Path(checkpoint_dir).mkdir(exist_ok=True)
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [train]", leave=False)
        for lr_batch, hr_batch in train_bar:
            lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)

            # Linear LR warmup over the first `warmup_steps` batches. This is
            # a from-scratch training run of a novel architecture (zero-init
            # residual + a 4-term composite loss) rather than a fine-tune of
            # an already-good checkpoint, so it's much more exposed to a
            # single early bad-gradient batch pushing the model into a
            # self-reinforcing bad state (pred error -> bigger pixel AND
            # perceptual loss -> bigger gradient in the same bad direction)
            # before the optimizer has any sense of the loss landscape.
            # Once warmup_steps is passed, the per-epoch CosineAnnealingLR
            # below takes back over -- this only touches the LR during the
            # warmup window.
            if global_step < warmup_steps:
                warmup_lr = lr * (global_step + 1) / warmup_steps
                for param_group in optimizer.param_groups:
                    param_group["lr"] = warmup_lr

            optimizer.zero_grad()
            # Loss deliberately computed outside autocast: SSIM's windowed
            # variances and NDVI's division are worth keeping in fp32, and
            # it's negligible next to the model forward. Inside the autocast
            # block, pred.float() would be undone anyway -- autocast re-casts
            # per-op, not per-input.
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                pred = model(lr_batch)
            loss = criterion(pred.float(), hr_batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            # clip_grad_norm_ combines every parameter's gradient into one
            # global norm and multiplies them all by a single clip_coef
            # derived from it -- so a non-finite norm (e.g. from one rare
            # extreme-but-finite gradient) poisons every parameter's update
            # in the same step. GradScaler's own inf/nan check runs during
            # unscale_() above, BEFORE this clipping call, so it can't catch
            # damage introduced here -- this is the gap that corrupted every
            # weight tensor in one shot during an earlier run. Skip the
            # update entirely rather than let scaler.step() apply it blind.
            if not torch.isfinite(grad_norm):
                print(f"\n  WARNING: non-finite grad norm ({grad_norm.item()}) at "
                      f"epoch {epoch+1} step {global_step} -- skipping this batch")
                optimizer.zero_grad()
                # scaler.update() must run even when skipping the step --
                # unscale_() marks this optimizer as "already unscaled" for
                # the current iteration, and only update() clears that mark.
                # Skip it and the NEXT iteration's unscale_() call raises
                # "unscale_() has already been called on this optimizer
                # since the last update()".
                scaler.update()
                global_step += 1
                continue

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}",
                                   lr=f"{optimizer.param_groups[0]['lr']:.2e}")
            global_step += 1
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [val]", leave=False)
        with torch.no_grad():
            for lr_batch, hr_batch in val_bar:
                lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    pred = model(lr_batch)
                batch_loss = criterion(pred.float(), hr_batch).item()
                val_loss += batch_loss
                val_bar.set_postfix(loss=f"{batch_loss:.4f}")
        val_loss /= len(val_loader)

        scheduler.step()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        gap_pct = ((val_loss - train_loss) / train_loss * 100) if train_loss > 0 else 0
        print(f"Epoch {epoch+1}/{num_epochs} | train: {train_loss:.4f} | "
              f"val: {val_loss:.4f} | gap: {gap_pct:+.1f}% | "
              f"lr: {scheduler.get_last_lr()[0]:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "num_blocks": num_blocks,
                "arch": "attn_rrdb",
            }, f"{checkpoint_dir}/best_model_attn.pt")
            print(f"  -> saved new best checkpoint (val_loss: {val_loss:.4f})")

        torch.save(model.state_dict(), f"{checkpoint_dir}/latest_model_attn.pt")

        should_stop, reason = early_stopper.check(epoch, train_loss, val_loss)
        if should_stop:
            print(f"\nStopping early at epoch {epoch+1}: {reason}")
            print(f"Best checkpoint retained at checkpoints/best_model_attn.pt "
                  f"(val_loss: {best_val_loss:.4f})")
            break
    else:
        print(f"\nCompleted all {num_epochs} epochs without triggering early stopping.")

    with open(f"{checkpoint_dir}/training_history_attn.json", "w") as f:
        json.dump(history, f)

    plot_loss_curve(history, f"{checkpoint_dir}/loss_curve_attn.png",
                     title="AttentionRRDBNet: Training vs Validation Loss")

    return history


if __name__ == "__main__":
    train()
