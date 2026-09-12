"""
train_attn_gan_v3_icnr: the attn_gan_v3 recipe, applied to an ICNR-initialised
base model.

Two stages, and this script is the second one. Stage 1 must be run first:

    python -m src.train_v2_icnr          # -> checkpoints/best_model_attn_icnr.pt
    python -m src.train_attn_gan_v3_icnr # -> checkpoints/best_model_attn_gan_v3_icnr.pt

Why this pairing. attn_gan_v3 is the best perceptual model so far (LPIPS 0.1091,
HF 0.780 against attn's 0.2219 / 0.532), but it carries a periodic mesh. Fourier
analysis located the cause: PixelShuffle builds each 2x2 output block from four
independently-initialised kernels that disagree from step zero, producing power
at period 2 and period 4 that the ground truth does not have --

    model            period-2   period-4
    ground truth         2.6x     151.0x
    attn                39.3x    2978.4x     <- present without any discriminator
    attn_gan           603.9x     417.0x
    attn_gan_v2        788.1x     625.3x

The artifact is architectural, not adversarial. Adversarial training does not
create it, it *unmasks* it by rewarding the high-frequency energy that fidelity
losses suppress -- which is why slowing the discriminator (v2) did not help.

ICNR initialises those four kernels as identical copies, so the upsampler starts
as exact nearest-neighbour resizing with zero checkerboard (verified: within-2x2
block spread 2.775 -> 0.000) and has to learn its way into any mesh rather than
starting with one. It only applies at initialisation, hence the retrain.

Everything else is held fixed at the settings that produced attn_gan_v3:
w_adv 5e-3 with lr_d 2e-5 -- v1's adversarial pressure with v2's slow
discriminator -- plus generator EMA at 0.999.

Two honest caveats:
  * ICNR fixes the starting point. Nothing in the loss penalises checkerboard,
    so whether it survives 100 epochs of training is untested. src/train_v2_icnr.py
    takes icnr_init=False, which makes a short A/B possible before committing.
  * Part of attn_gan_v3's 0.780 HF ratio is mesh rather than recovered detail,
    since the metric cannot tell them apart. If this run's HF comes in LOWER
    while LPIPS also improves, that is success, not regression -- it means the
    artifact was removed and the remaining sharpness is real.

Usage:
    python -m src.train_attn_gan_v3_icnr
"""

from pathlib import Path

from src.train_v2_gan import train

ICNR_BASE = "checkpoints/best_model_attn_icnr.pt"

# attn_gan_v3's settings, unchanged -- only the base model differs
V3_SETTINGS = dict(
    w_adv=5e-3,        # v1's adversarial pressure: what produced the sharpness
    lr_d=2e-5,         # v2's slow discriminator: what prevented the collapse
    ema_decay=0.999,   # smooths the oscillation adversarial training always has
    num_epochs=40,
)


def main():
    if not Path(ICNR_BASE).exists():
        raise SystemExit(
            f"{ICNR_BASE} not found.\n\n"
            f"This is stage 2 of 2. Train the ICNR base first:\n"
            f"    python -m src.train_v2_icnr\n\n"
            f"Or, to check ICNR survives training before committing ~4 hours to it,\n"
            f"run the 8-epoch A/B first:\n"
            f"    python -c \"from src.train_v2_icnr import train; "
            f"train(num_epochs=8, icnr_init=True,  run_suffix='_icnr_ab_on')\"\n"
            f"    python -c \"from src.train_v2_icnr import train; "
            f"train(num_epochs=8, icnr_init=False, run_suffix='_icnr_ab_off')\"\n"
            f"    python -m src.eval.checkerboard --compare")

    print("Stage 2/2 · adversarial refinement of the ICNR base")
    print(f"  base:     {ICNR_BASE}")
    print(f"  settings: {V3_SETTINGS}  (identical to attn_gan_v3)")
    print(f"  target:   beat attn_gan_v3 -- LPIPS 0.1091, and reduce period-2/4 power")
    print(f"            (a LOWER HF ratio alongside a lower LPIPS means the mesh went,")
    print(f"             not the detail)\n")

    return train(warm_start=ICNR_BASE, run_suffix="_v3_icnr", **V3_SETTINGS)


if __name__ == "__main__":
    main()
