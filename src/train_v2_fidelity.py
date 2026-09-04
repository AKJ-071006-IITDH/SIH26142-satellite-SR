"""
train_v2_fidelity: pixel-fidelity-weighted variant of train_v2.py's
AttentionRRDBNet training run.

Same architecture, same warmup/LR/grad-clip fix -- only two things change:

1. Loss weights shift toward pixel accuracy. The proven attn checkpoint
   (checkpoints/best_model_attn.pt, 37.28 dB / 0.878 SSIM on the full test
   split) showed SSIM+spectral making up ~82% of its remaining loss late in
   training, with pixel down at 13% -- most of the gradient signal was
   chasing structure/NDVI agreement, not pixel closeness to ground truth.
   w_pixel 0.8->1.0, w_ssim 0.3->0.2, w_spectral 0.5->0.2 (w_perceptual
   unchanged -- it wasn't flagged as oversized).
2. dropout_rate 0.1->0.05. The 0.1 default was sized for the old, smaller
   (1,469-patch) dataset; the expanded 3,264-patch set can likely support
   less regularization.

Whether this actually beats the existing checkpoint is exactly what this
run is testing -- it's an experiment, not an assumed improvement, which is
why it writes to its own checkpoint files (*_fidelity suffix) rather than
touching checkpoints/best_model_attn.pt.

Usage:
    python -m src.train_v2_fidelity
"""

import torch
from pathlib import Path
from tqdm import tqdm

from src.models.rrdbnet_attn import AttentionRRDBNet
from src.losses.losses_attn import CombinedFidelityLossV2
from src.eval.plot_history import plot_loss_curve
from src.train_v2 import get_dataloaders, EarlyStopper, build_feature_extractor


def train(num_epochs=100, lr=1e-4, batch_size=8, checkpoint_dir="checkpoints",
          num_blocks=8, patience=7, overfit_gap_threshold=0.15,
          warmup_steps=300, grad_clip_norm=0.5, dropout_rate=0.05,
          w_pixel=1.0, w_ssim=0.2, w_spectral=0.2, w_perceptual=0.3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training AttentionRRDBNet [fidelity-weighted] on: {device}")
    print(f"  loss weights: pixel={w_pixel} ssim={w_ssim} spectral={w_spectral} "
          f"perceptual={w_perceptual} | dropout_rate={dropout_rate}")

    train_loader, val_loader = get_dataloaders(batch_size)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = AttentionRRDBNet(in_channels=4, out_channels=4, num_blocks=num_blocks,
                              scale_factor=4, dropout_rate=dropout_rate).to(device)

    feature_extractor = build_feature_extractor(device)
    criterion = CombinedFidelityLossV2(feature_extractor=feature_extractor,
                                        w_pixel=w_pixel, w_ssim=w_ssim,
                                        w_spectral=w_spectral, w_perceptual=w_perceptual)

    frozen = [name for name, p in model.named_parameters() if not p.requires_grad]
    assert not frozen, (f"{len(frozen)} model parameters are frozen before training "
                        f"(first: {frozen[0]}) -- the loss is holding a reference to "
                        f"the model being trained.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # bf16 rather than fp16 -- see train_v2.py for the rationale. It matters
    # more here: this config's gradients measure ~2.2x larger than train_v2.py's
    # (median norm 0.254 vs 0.114), leaving only ~1.45x headroom below fp16's
    # overflow ceiling versus 2.5x, so outlier batches tipped it over
    # repeatedly, collapsed the loss scale, and degraded training.
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

            if global_step < warmup_steps:
                warmup_lr = lr * (global_step + 1) / warmup_steps
                for param_group in optimizer.param_groups:
                    param_group["lr"] = warmup_lr

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                pred = model(lr_batch)
            loss = criterion(pred.float(), hr_batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            # See train_v2.py for why this check exists: clip_grad_norm_
            # combines every parameter into one global norm and rescales
            # them all by a single clip_coef derived from it, so a
            # non-finite norm poisons every parameter's update in one shot
            # -- and GradScaler can't catch it, since its own inf/nan check
            # runs during unscale_() above, before this call. This exact gap
            # is what corrupted every weight tensor in a previous run of
            # this script (all 396/396 tensors went NaN in a single step).
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
                "dropout_rate": dropout_rate,
                "arch": "attn_rrdb",
                "loss_weights": {"pixel": w_pixel, "ssim": w_ssim,
                                  "spectral": w_spectral, "perceptual": w_perceptual},
            }, f"{checkpoint_dir}/best_model_attn_fidelity.pt")
            print(f"  -> saved new best checkpoint (val_loss: {val_loss:.4f})")

        torch.save(model.state_dict(), f"{checkpoint_dir}/latest_model_attn_fidelity.pt")

        should_stop, reason = early_stopper.check(epoch, train_loss, val_loss)
        if should_stop:
            print(f"\nStopping early at epoch {epoch+1}: {reason}")
            print(f"Best checkpoint retained at checkpoints/best_model_attn_fidelity.pt "
                  f"(val_loss: {best_val_loss:.4f})")
            break
    else:
        print(f"\nCompleted all {num_epochs} epochs without triggering early stopping.")

    with open(f"{checkpoint_dir}/training_history_attn_fidelity.json", "w") as f:
        import json
        json.dump(history, f)

    plot_loss_curve(history, f"{checkpoint_dir}/loss_curve_attn_fidelity.png",
                     title="AttentionRRDBNet [fidelity-weighted]: Training vs Validation Loss")

    return history


if __name__ == "__main__":
    train()
