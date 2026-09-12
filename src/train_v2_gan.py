"""
train_v2_gan: adversarial refinement of the attn checkpoint, aimed at visual
resemblance to the HR ground truth rather than at distortion metrics.

Why this exists. The fidelity tracks plateaued: attn 37.282 dB, attn_fidelity
37.019, and reweighting the same four fidelity terms moved nothing meaningful.
That is not a tuning failure -- it is the objective working as designed. Every
term in those losses is minimised by predicting the conditional mean, and the
conditional mean of "roof edge or shadow?" is grey mush. The outputs are
statistically close and visibly smooth.

A discriminator is the one signal that penalises that hedge, because a blurry
patch is trivially identifiable as fake regardless of how low its MSE is.

Expect PSNR to DROP, roughly 0.5-1 dB, while the images look better. That is
the perception-distortion tradeoff, and it is a proven result rather than a
bug. best_model_attn.pt stays untouched as the distortion champion; this run
produces a separate perceptual checkpoint.

Two things this run does that the repo's earlier GAN attempt (train_phase4.py)
did not:

  * The frozen feature extractor is a genuinely separate network (VGG19), not
    the live generator. train_phase4.py passes its own generator to
    SpectralPerceptualLoss, which sets requires_grad=False on the shared
    modules and silently freezes 46% of the generator -- including conv_first
    -- for the whole run. That is very likely why the Phase 3/4 results
    disappointed, and it means "GANs don't help here" was never actually
    tested on this data.
  * Checkpoints are selected on LPIPS, not on fidelity loss. Selecting on
    fidelity loss during adversarial training picks the LEAST sharp model,
    which is exactly backwards for this goal.

Usage:
    python -m src.train_v2_gan
    python -m src.eval.perceptual_metrics --compare      # judge the result
"""

import json

import torch
from pathlib import Path
from tqdm import tqdm

from src.models.rrdbnet_attn import AttentionRRDBNet
from src.models.discriminator import PatchDiscriminator
from src.losses.losses_gan import VGGPerceptualLoss, GeneratorLoss, DiscriminatorLoss
from src.eval.plot_history import plot_loss_curve
from src.eval.perceptual_metrics import LPIPSScorer, hf_energy_ratio, build_loader
from src.eval.metrics import evaluate_batch
from src.train_v2 import get_dataloaders


class EMA:
    """
    Exponential moving average of the generator weights.

    Adversarial training does not converge, it oscillates -- v1's LPIPS went
    0.119, 0.131, 0.123, 0.133 on consecutive epochs, so which checkpoint you
    keep is partly luck of the draw. An EMA of the weights sits nearer the
    centre of that oscillation than any single point on it, which is why it is
    standard practice for GANs. Evaluate and checkpoint the averaged weights,
    keep training the live ones.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)

    def copy_to(self, model):
        """Load averaged weights into `model`, returning the originals to restore."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()
                  if k in self.shadow}
        model.load_state_dict({k: v.to(dtype=model.state_dict()[k].dtype)
                               for k, v in self.shadow.items()}, strict=False)
        return backup

    def restore(self, model, backup):
        model.load_state_dict(backup, strict=False)


def evaluate_perceptual(generator, loader, lpips_scorer, device, amp_dtype, use_amp):
    """Per-epoch scoring on the metrics that actually match the goal."""
    generator.eval()
    psnr, ssim, lp, hf = [], [], [], []
    with torch.no_grad():
        for lr_b, hr_b in loader:
            lr_b, hr_b = lr_b.to(device), hr_b.to(device)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                pred = generator(lr_b)
            pred = pred.float()
            m = evaluate_batch(pred, hr_b, scale_factor=4)
            psnr.append(m["psnr"]); ssim.append(m["ssim"])
            hf.append(hf_energy_ratio(pred.clamp(0, 1), hr_b))
            if lpips_scorer.available:
                lp.append(lpips_scorer(pred, hr_b))
    generator.train()
    mean = lambda v: (sum(v) / len(v)) if v else float("nan")
    return mean(psnr), mean(ssim), mean(lp), mean(hf)


def train(num_epochs=40, lr_g=5e-5, lr_d=1e-4, batch_size=8,
          checkpoint_dir="checkpoints", warmup_steps=200, grad_clip_norm=0.5,
          w_pixel=1.0, w_ssim=0.15, w_spectral=0.3, w_vgg=0.1, w_adv=5e-3,
          eval_patches=96, save_every=5, run_suffix="", ema_decay=0.999,
          warm_start="checkpoints/best_model_attn.pt"):
    """
    run_suffix is appended to every output filename, so a re-run with different
    hyperparameters cannot clobber a previous run's checkpoint. Adversarial runs
    are volatile and the good checkpoint often arrives early -- losing one to an
    experiment that later destabilised would be an expensive mistake.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = f"attn_gan{run_suffix}"
    print(f"Training AttentionRRDBNet [adversarial refinement] on: {device}")
    print(f"  generator loss: pixel={w_pixel} ssim={w_ssim} spectral={w_spectral} "
          f"vgg={w_vgg} adv={w_adv}")
    print(f"  lr_g={lr_g}  lr_d={lr_d}")
    print(f"  writing to: checkpoints/best_model_{tag}.pt")

    train_loader, _ = get_dataloaders(batch_size)
    print(f"Train batches: {len(train_loader)}")

    # --- generator, warm-started from the fidelity champion ---
    if not Path(warm_start).exists():
        raise FileNotFoundError(
            f"{warm_start} not found. Adversarial refinement fine-tunes an already-good "
            f"model; train src/train_v2.py first.")
    ckpt = torch.load(warm_start, map_location=device)
    generator = AttentionRRDBNet(in_channels=4, out_channels=4,
                                  num_blocks=ckpt["num_blocks"], scale_factor=4,
                                  dropout_rate=ckpt.get("dropout_rate", 0.1)).to(device)
    generator.load_state_dict(ckpt["model_state_dict"])
    print(f"Warm-started generator from {warm_start} "
          f"(epoch {ckpt['epoch']+1}, val_loss {ckpt['val_loss']:.4f})")

    ema = EMA(generator, decay=ema_decay) if ema_decay else None
    if ema:
        print(f"Generator EMA enabled (decay={ema_decay}) -- scoring and checkpointing"
              f" the averaged weights")

    discriminator = PatchDiscriminator(in_channels=4).to(device)

    # VGG19 is a separate frozen network -- never the live generator. See the
    # module docstring: passing the trained model to a loss that freezes what
    # it is handed silently freezes the model itself.
    vgg = VGGPerceptualLoss().to(device)
    print("Perceptual features: frozen VGG19 conv5_4 (ImageNet), RGB bands only")

    frozen = [n for n, p in generator.named_parameters() if not p.requires_grad]
    assert not frozen, (f"{len(frozen)} generator parameters are frozen before training "
                        f"(first: {frozen[0]}) -- a loss is holding a reference to the "
                        f"model being trained.")

    criterion_g = GeneratorLoss(vgg_extractor=vgg, w_pixel=w_pixel, w_ssim=w_ssim,
                                 w_spectral=w_spectral, w_vgg=w_vgg, w_adv=w_adv)
    criterion_d = DiscriminatorLoss()

    opt_g = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.9, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.9, 0.999))
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=num_epochs)
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=num_epochs)

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    needs_scaler = use_amp and amp_dtype is torch.float16
    scaler_g = torch.amp.GradScaler("cuda", enabled=needs_scaler)
    scaler_d = torch.amp.GradScaler("cuda", enabled=needs_scaler)
    if use_amp:
        print(f"AMP dtype: {amp_dtype} (GradScaler {'on' if needs_scaler else 'off -- not needed for bf16'})")

    eval_loader, n_eval = build_loader("val", limit=eval_patches, batch_size=batch_size)
    lpips_scorer = LPIPSScorer(net="alex", device=device)
    if lpips_scorer.available:
        print(f"Selecting checkpoints on LPIPS (alex) over {n_eval} val patches")
    else:
        print(f"NOTE: {lpips_scorer.error}")
        print("      Falling back to HF energy ratio for checkpoint selection.")

    Path(checkpoint_dir).mkdir(exist_ok=True)
    best_score = float("inf")
    history = {"train_loss": [], "val_loss": [], "psnr": [], "ssim": [],
               "lpips": [], "hf_ratio": [], "d_loss": []}
    global_step = 0

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        g_running, d_running = 0.0, 0.0
        parts_running = {}
        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [train]", leave=False)

        for lr_batch, hr_batch in bar:
            lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)

            if global_step < warmup_steps:
                for pg in opt_g.param_groups:
                    pg["lr"] = lr_g * (global_step + 1) / warmup_steps

            # ---- discriminator ----
            opt_d.zero_grad()
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                fake = generator(lr_batch)
            fake_det = fake.detach().float()
            loss_d = criterion_d(discriminator(hr_batch), discriminator(fake_det))
            scaler_d.scale(loss_d).backward()
            scaler_d.unscale_(opt_d)
            dn = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip_norm)
            if torch.isfinite(dn):
                scaler_d.step(opt_d)
            scaler_d.update()

            # ---- generator ----
            opt_g.zero_grad()
            pred_fake = discriminator(fake.float())
            loss_g, parts = criterion_g(fake.float(), hr_batch, pred_fake)
            scaler_g.scale(loss_g).backward()
            scaler_g.unscale_(opt_g)
            gn = torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip_norm)
            if not torch.isfinite(gn):
                print(f"\n  WARNING: non-finite generator grad norm at epoch {epoch+1} "
                      f"step {global_step} -- skipping this batch")
                opt_g.zero_grad()
                scaler_g.update()
                global_step += 1
                continue
            scaler_g.step(opt_g)
            scaler_g.update()
            if ema:
                ema.update(generator)

            g_running += loss_g.item()
            d_running += loss_d.item()
            for k, v in parts.items():
                parts_running[k] = parts_running.get(k, 0.0) + v
            bar.set_postfix(G=f"{loss_g.item():.4f}", D=f"{loss_d.item():.4f}")
            global_step += 1

        n_batches = len(train_loader)
        g_running /= n_batches
        d_running /= n_batches
        parts_running = {k: v / n_batches for k, v in parts_running.items()}

        sched_g.step()
        sched_d.step()

        backup = ema.copy_to(generator) if ema else None
        psnr, ssim, lpips_val, hf = evaluate_perceptual(
            generator, eval_loader, lpips_scorer, device, amp_dtype, use_amp)

        history["train_loss"].append(g_running)
        history["val_loss"].append(g_running)   # plot_loss_curve expects both keys
        history["d_loss"].append(d_running)
        history["psnr"].append(psnr)
        history["ssim"].append(ssim)
        history["lpips"].append(lpips_val)
        history["hf_ratio"].append(hf)

        parts_str = " ".join(f"{k}:{v:.4f}" for k, v in parts_running.items())
        lp_str = f"{lpips_val:.4f}" if lpips_scorer.available else "n/a"
        print(f"Epoch {epoch+1}/{num_epochs} | G: {g_running:.4f} | D: {d_running:.4f} | "
              f"{parts_str}")
        print(f"           PSNR {psnr:.3f} | SSIM {ssim:.4f} | LPIPS {lp_str} | "
              f"HF {hf:.3f}  (HF 1.000 = as sharp as ground truth)")

        # Selection metric: LPIPS if available, else distance of HF ratio from 1.0.
        # Deliberately NOT fidelity loss -- that would pick the blurriest model.
        current = lpips_val if lpips_scorer.available else abs(hf - 1.0)
        if current < best_score:
            best_score = current
            torch.save({
                "epoch": epoch,
                "model_state_dict": generator.state_dict(),
                "num_blocks": ckpt["num_blocks"],
                "dropout_rate": ckpt.get("dropout_rate", 0.1),
                "arch": "attn_rrdb",
                "val_loss": g_running,
                "psnr": psnr, "ssim": ssim, "lpips": lpips_val, "hf_ratio": hf,
                "selected_on": "lpips" if lpips_scorer.available else "hf_ratio",
            }, f"{checkpoint_dir}/best_model_{tag}.pt")
            print(f"  -> new best perceptual checkpoint "
                  f"({'LPIPS' if lpips_scorer.available else 'HF'} {current:.4f})")

        if (epoch + 1) % save_every == 0:
            torch.save({"epoch": epoch, "model_state_dict": generator.state_dict(),
                        "num_blocks": ckpt["num_blocks"], "arch": "attn_rrdb",
                        "psnr": psnr, "lpips": lpips_val, "hf_ratio": hf},
                       f"{checkpoint_dir}/{tag}_epoch{epoch+1:03d}.pt")

        # hand the live weights back -- everything above scored and saved the
        # EMA copy, and training must continue from the real ones
        if ema:
            ema.restore(generator, backup)

    with open(f"{checkpoint_dir}/training_history_{tag}.json", "w") as f:
        json.dump(history, f)
    plot_loss_curve(history, f"{checkpoint_dir}/loss_curve_{tag}.png",
                     title="AttentionRRDBNet [adversarial]: Generator Loss")
    print("\nDone. Compare against the fidelity models with:")
    print("  python -m src.eval.perceptual_metrics --compare")
    return history


if __name__ == "__main__":
    train()
