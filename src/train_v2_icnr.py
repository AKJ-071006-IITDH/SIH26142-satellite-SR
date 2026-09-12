"""
train_v2_icnr: retrains AttentionRRDBNet from scratch with ICNR initialisation
on the pixel-shuffle upsamplers.

Why. Fourier analysis of the existing checkpoints found a periodic component
that the ground truth does not have, at exactly the two frequencies
PixelShuffle produces (period 4 from the first x2 stage, doubled by the
second; period 2 from the second stage):

    model              period-2 peak    period-4 peak
    ground truth              1.9x            57.6x
    attn                     57.6x          3677.0x     <- no discriminator, ever
    attn_gan                119.0x           520.0x
    attn_gan_v2             154.8x           332.8x

The artifact is therefore architectural, not adversarial. attn has it without
ever meeting a discriminator; adversarial training merely *unmasks* it by
rewarding the high-frequency energy that fidelity losses suppress. That is
also why slowing the discriminator (attn_gan_v2) did not remove it.

ICNR initialises the r^2 sub-pixel kernels of each upsampler as identical
copies, so the upsampler starts as exact nearest-neighbour resizing with zero
checkerboard -- verified: within-2x2-block spread goes from 2.775 to exactly
0.000. Training then has to learn its way into any checkerboard instead of
starting with one.

This is a SINGLE-VARIABLE experiment against attn. Loss weights, dropout,
learning rate, warmup, precision, clipping, batch size and dataset are all
identical to train_v2.py -- only icnr_init changes. If the period-4 excess
drops relative to attn, the diagnosis was right.

Because ICNR only applies at initialisation, this cannot be retrofitted onto
best_model_attn.pt; it requires training from scratch. Afterwards, refine it
adversarially with:

    python -c "from src.train_v2_gan import train; \\
               train(warm_start='checkpoints/best_model_attn_icnr.pt', run_suffix='_icnr')"

Usage:
    python -m src.train_v2_icnr
"""

import json

import torch
from pathlib import Path
from tqdm import tqdm

from src.models.rrdbnet_attn import AttentionRRDBNet
from src.losses.losses_attn import CombinedFidelityLossV2
from src.eval.plot_history import plot_loss_curve
from src.train_v2 import get_dataloaders, EarlyStopper, build_feature_extractor


def train(num_epochs=100, lr=1e-4, batch_size=8, checkpoint_dir="checkpoints",
          num_blocks=8, patience=7, overfit_gap_threshold=0.15,
          warmup_steps=300, grad_clip_norm=0.5, dropout_rate=0.1,
          w_pixel=0.8, w_ssim=0.3, w_spectral=0.5, w_perceptual=0.3,
          run_suffix="_icnr", icnr_init=True):
    """
    icnr_init is exposed so this same script can run its own control. Setting
    it False reproduces train_v2.py exactly, which makes a short A/B possible:
    run a few epochs each way and compare src.eval.checkerboard, rather than
    spending a full run on the assumption that the fix holds through training.
    ICNR only sets the starting point -- nothing in the loss penalises
    checkerboard, so whether it persists is an empirical question.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = f"attn{run_suffix}"
    mode = "ICNR upsampler init" if icnr_init else "CONTROL - default init"
    print(f"Training AttentionRRDBNet [{mode}] on: {device}")
    print(f"  loss weights: pixel={w_pixel} ssim={w_ssim} spectral={w_spectral} "
          f"perceptual={w_perceptual} | dropout_rate={dropout_rate}")
    print(f"  identical to train_v2.py except icnr_init={icnr_init} -- single-variable test")
    print(f"  writing to: {checkpoint_dir}/best_model_{tag}.pt")

    train_loader, val_loader = get_dataloaders(batch_size)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = AttentionRRDBNet(in_channels=4, out_channels=4, num_blocks=num_blocks,
                              scale_factor=4, dropout_rate=dropout_rate,
                              icnr_init=icnr_init).to(device)

    # confirm the initialisation actually took: every 2x2 output block of each
    # upsampler must be uniform, or the whole premise of this run is wrong
    with torch.no_grad():
        probe = model.upsample[0](torch.randn(1, 64, 8, 8, device=device))
        b = probe.reshape(1, 64, 4, 2, 4, 2)
        spread = (b.amax(dim=(3, 5)) - b.amin(dim=(3, 5))).max().item()
    print(f"  init check: max within-2x2-block spread = {spread:.2e} "
          f"({'0 = exact nearest-neighbour' if icnr_init else 'nonzero = checkerboard present, as expected for the control'})")
    if icnr_init:
        assert spread < 1e-5, "ICNR initialisation did not take effect"

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

            # See train_v2.py: clip_grad_norm_ rescales every parameter by one
            # global coefficient, so a non-finite norm poisons the whole model
            # in a single step, and GradScaler's check runs before this call.
            if not torch.isfinite(grad_norm):
                print(f"\n  WARNING: non-finite grad norm ({grad_norm.item()}) at "
                      f"epoch {epoch+1} step {global_step} -- skipping this batch")
                optimizer.zero_grad()
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
                "icnr_init": True,
                "loss_weights": {"pixel": w_pixel, "ssim": w_ssim,
                                  "spectral": w_spectral, "perceptual": w_perceptual},
            }, f"{checkpoint_dir}/best_model_{tag}.pt")
            print(f"  -> saved new best checkpoint (val_loss: {val_loss:.4f})")

        torch.save(model.state_dict(), f"{checkpoint_dir}/latest_model_{tag}.pt")

        should_stop, reason = early_stopper.check(epoch, train_loss, val_loss)
        if should_stop:
            print(f"\nStopping early at epoch {epoch+1}: {reason}")
            print(f"Best checkpoint retained at {checkpoint_dir}/best_model_{tag}.pt "
                  f"(val_loss: {best_val_loss:.4f})")
            break
    else:
        print(f"\nCompleted all {num_epochs} epochs without triggering early stopping.")

    with open(f"{checkpoint_dir}/training_history_{tag}.json", "w") as f:
        json.dump(history, f)

    plot_loss_curve(history, f"{checkpoint_dir}/loss_curve_{tag}.png",
                     title="AttentionRRDBNet [ICNR init]: Training vs Validation Loss")

    print("\nDone. Check whether the checkerboard actually dropped:")
    print("  python -m src.eval.checkerboard --compare")
    return history


if __name__ == "__main__":
    train()
