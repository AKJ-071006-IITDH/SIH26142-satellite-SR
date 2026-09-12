"""
train_v3: AttentionRRDBNet trained on random crops from the raw tiles, using
the leak-free manifest split.

Single-variable test against train_v2.py. Architecture, loss weights, learning
rate, warmup, precision, clipping and batch size are all identical -- only the
data pipeline changes:

    train_v2.py   2,449 pre-cut 128x128 patches, patch-level random split.
                  Every epoch sees the same 2,449 crops. Train and test crops
                  come from the same tiles: measured 326/326 test patches had
                  a train patch directly adjacent.

    train_v3.py   random crops sampled from 36 whole tiles (6 regions), with
                  goa_coastal held out for val and chennai_urban +
                  deccan_plateau held out for test. No tile appears in more
                  than one split. Each epoch draws fresh crops, so over a run
                  the model sees ~200k distinct crops rather than 2,449.

The leak itself turned out to be worth only ~0.2 dB, so this run is not about
fixing that -- it is about the training-data diversity, which is the one
substantial lever left untested. The model has never seen a crop outside those
original 2,449.

Which split to use, and why spatial rather than region:

    manifest_spatial.json  (default) -- all 9 regions and all 6 seasons appear
        in train, val AND test, as disjoint bands with a 64 px guard between
        them. This matters most for VAL: val_loss drives early stopping and
        checkpoint selection, so a val set covering one land-cover type would
        select whichever checkpoint happened to suit that type. Region mode put
        only goa_coastal in val -- the easiest region in the set, ~13 dB above
        chennai_urban -- which would have biased selection toward smooth water.

    manifest_region.json -- whole regions held out; test geography is entirely
        unseen. Stricter, but val/test then cover only the held-out land-cover
        types. Useful as a secondary generalisation check, not for selection.

The residual leak in spatial mode is bounded by measurement: directly adjacent
train/test crops were worth only ~0.2 dB on this data, and banding with a guard
is strictly weaker than adjacency. Representative validation is worth that.

val_loss is NOT comparable to train_v2.py's -- different val distribution.
Judge this run on the test table, not the training log.

Prerequisite:
    python -m src.data.spatial_splits --mode spatial --out data/splits/manifest_spatial.json

Usage:
    python -m src.train_v3
    python -m src.eval.perceptual_metrics --manifest data/splits/manifest_spatial.json --compare
"""

import json

import torch
from pathlib import Path
from tqdm import tqdm

from src.models.rrdbnet_attn import AttentionRRDBNet
from src.losses.losses_attn import CombinedFidelityLossV2
from src.data.tile_dataset import get_tile_dataloaders
from src.eval.plot_history import plot_loss_curve
from src.train_v2 import EarlyStopper, build_feature_extractor


def train(num_epochs=100, lr=1e-4, batch_size=8, checkpoint_dir="checkpoints",
          num_blocks=8, patience=7, overfit_gap_threshold=0.15,
          warmup_steps=300, grad_clip_norm=0.5, dropout_rate=0.1,
          w_pixel=0.8, w_ssim=0.3, w_spectral=0.5, w_perceptual=0.3,
          manifest="data/splits/manifest_spatial.json", train_length=2456,
          icnr_init=False, run_suffix="_v3"):
    """
    train_length defaults to 2,456 (= 307 batches of 8) so an epoch costs the
    same number of optimiser steps as train_v2.py's did. With random cropping
    "epoch" is otherwise an arbitrary unit, and matching it keeps the cosine
    schedule and early-stopping behaviour comparable between the two runs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = f"attn{run_suffix}"

    if not Path(manifest).exists():
        raise FileNotFoundError(
            f"{manifest} not found. Build it first:\n"
            f"    python -m src.data.spatial_splits")
    with open(manifest) as f:
        mf = json.load(f)

    print(f"Training AttentionRRDBNet [random crops, leak-free split] on: {device}")
    print(f"  manifest: {manifest} (mode={mf['mode']})")
    print(f"  loss weights: pixel={w_pixel} ssim={w_ssim} spectral={w_spectral} "
          f"perceptual={w_perceptual} | dropout={dropout_rate} | icnr={icnr_init}")
    print(f"  identical to train_v2.py except the data pipeline -- single-variable test")
    print(f"  writing to: {checkpoint_dir}/best_model_{tag}.pt")

    train_loader, val_loader = get_tile_dataloaders(
        batch_size=batch_size, manifest_path=manifest, train_length=train_length)
    print(f"Train batches: {len(train_loader)} (fresh random crops each epoch) | "
          f"Val batches: {len(val_loader)} (fixed grid)")

    model = AttentionRRDBNet(in_channels=4, out_channels=4, num_blocks=num_blocks,
                              scale_factor=4, dropout_rate=dropout_rate,
                              icnr_init=icnr_init).to(device)

    # NOTE: the perceptual extractor is the Phase 1 RRDBNet, which was trained
    # on the old split and has therefore seen the held-out regions. It is a
    # frozen feature space, not a predictor -- the same relationship an
    # ImageNet-pretrained VGG has to any dataset -- so it supplies no target
    # information. Worth knowing about, not a blocker.
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
            # in one step, and GradScaler's check runs before this call.
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
                "icnr_init": icnr_init,
                "data_pipeline": "random_crop_manifest",
                "manifest_mode": mf["mode"],
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
                     title="AttentionRRDBNet [random crops]: Training vs Validation Loss")

    print("\nDone. Evaluate on the leak-free test split:")
    print("  python -m src.eval.perceptual_metrics --manifest --compare")
    return history


if __name__ == "__main__":
    train()
