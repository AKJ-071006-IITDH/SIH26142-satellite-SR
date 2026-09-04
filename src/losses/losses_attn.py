import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.losses import CharbonnierLoss, SpectralConsistencyLoss, SpectralPerceptualLoss

# src/losses/losses_attn.py — new loss pieces for the attention-RRDB track.
#
# CharbonnierLoss / SpectralConsistencyLoss / SpectralPerceptualLoss are
# architecture-agnostic and already solid, so they're reused (imported) from
# src/losses/losses.py rather than duplicated -- only the genuinely new piece
# (SSIM) and its composition live here.


def _gaussian_window(window_size, sigma, channels, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(1)
    window_2d = g @ g.t()
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


class SSIMLoss(nn.Module):
    """
    Differentiable windowed SSIM, returned as (1 - SSIM) so it's minimized
    like the other loss terms. Hand-rolled rather than pulled from
    scikit-image (used for the non-differentiable eval-time SSIM in
    src/eval/metrics.py) since that implementation can't backprop through,
    and rather than adding a new pytorch-msssim dependency for one small
    formula.

    SSIM rewards matching local structure (luminance/contrast/structure),
    not just per-pixel closeness -- a useful complement to the Charbonnier
    pixel loss when the goal is fidelity to ground truth, not just low
    average error.
    """
    def __init__(self, window_size=11, sigma=1.5, data_range=1.0):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.C1 = (0.01 * data_range) ** 2
        self.C2 = (0.03 * data_range) ** 2

    def forward(self, pred, target):
        channels = pred.shape[1]
        window = _gaussian_window(self.window_size, self.sigma, channels,
                                   pred.device, pred.dtype)
        pad = self.window_size // 2

        mu_pred = F.conv2d(pred, window, padding=pad, groups=channels)
        mu_target = F.conv2d(target, window, padding=pad, groups=channels)

        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target

        var_pred = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu_pred_sq
        var_target = F.conv2d(target * target, window, padding=pad, groups=channels) - mu_target_sq
        covar = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu_pred_target

        ssim_map = ((2 * mu_pred_target + self.C1) * (2 * covar + self.C2)) / \
                   ((mu_pred_sq + mu_target_sq + self.C1) * (var_pred + var_target + self.C2))

        return 1 - ssim_map.mean()


class CombinedFidelityLossV2(nn.Module):
    def __init__(self, feature_extractor=None,
                 w_pixel=0.8, w_ssim=0.3, w_spectral=0.5, w_perceptual=0.3):
        """
        Purely fidelity-driven -- no adversarial term. Weights kept
        pixel-dominant, same philosophy as CombinedSRLoss in losses.py:
        pixel loss is the most stable signal, SSIM/spectral/perceptual are
        refinements on top rather than the primary target.
        """
        super().__init__()
        self.pixel_loss = CharbonnierLoss()
        self.ssim_loss = SSIMLoss()
        self.spectral_loss = SpectralConsistencyLoss()
        self.perceptual_loss = SpectralPerceptualLoss(feature_extractor) if feature_extractor else None
        self.w_pixel = w_pixel
        self.w_ssim = w_ssim
        self.w_spectral = w_spectral
        self.w_perceptual = w_perceptual

    def forward(self, pred, target):
        loss = self.w_pixel * self.pixel_loss(pred, target)
        loss = loss + self.w_ssim * self.ssim_loss(pred, target)
        loss = loss + self.w_spectral * self.spectral_loss(pred, target)
        if self.perceptual_loss is not None:
            loss = loss + self.w_perceptual * self.perceptual_loss(pred, target)
        return loss
