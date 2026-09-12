import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from src.losses.losses import CharbonnierLoss, SpectralConsistencyLoss
from src.losses.losses_attn import SSIMLoss

# src/losses/losses_gan.py -- objective for adversarial refinement.
#
# The fidelity-only tracks (train_v2.py, train_v2_fidelity*.py) all optimise
# variations on "be close in value to the target". Every one of those terms is
# minimised by hedging: where the model cannot tell roof edge from shadow, the
# lowest-error answer is something in between, and averaged over a patch that
# is blur. This file adds the two signals that penalise hedging directly.


class VGGPerceptualLoss(nn.Module):
    """
    L1 distance between VGG19 features of prediction and target.

    Replaces the self-feature extractor used in the fidelity tracks. That one
    compared features from the project's own RRDBNet -- but those layers were
    themselves trained with fidelity losses, so they encode the same
    blur-tolerant statistics and cannot object to blur the pixel loss already
    accepts. VGG19 was trained discriminatively on ImageNet, so its features
    genuinely separate "sharp texture" from "smooth average".

    Uses conv5_4 pre-activation (feature index 34), the ESRGAN convention --
    pre-activation features are denser and carry more signal than the sparse
    post-ReLU ones.

    Input is 4-band reflectance; VGG wants 3-channel ImageNet-normalised RGB,
    so the first three bands are taken and normalised. The NIR band is left to
    the pixel and NDVI terms, which do see all four.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, feature_index=34):
        super().__init__()
        weights = torchvision.models.VGG19_Weights.IMAGENET1K_V1
        vgg = torchvision.models.vgg19(weights=weights)
        self.features = vgg.features[:feature_index + 1].eval()
        for p in self.features.parameters():
            p.requires_grad = False

        self.register_buffer("mean", torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1))

    def _prep(self, x):
        rgb = x[:, :3].clamp(0, 1)
        return (rgb - self.mean) / self.std

    def forward(self, pred, target):
        return F.l1_loss(self.features(self._prep(pred)),
                          self.features(self._prep(target)))


class GeneratorLoss(nn.Module):
    """
    Fidelity anchor + VGG perceptual + adversarial.

    The fidelity terms are kept (at reduced weight) rather than dropped: they
    are what stops the discriminator from talking the generator into inventing
    texture that is plausible but not *this* scene. Their job here is to hold
    the output anchored to the actual ground truth while the adversarial term
    supplies sharpness.

    w_adv is deliberately small. The adversarial gradient is the least
    trustworthy signal in the stack -- large values buy sharpness by
    hallucinating detail, which for satellite imagery means inventing
    buildings and field boundaries that are not there.
    """

    def __init__(self, vgg_extractor=None, w_pixel=1.0, w_ssim=0.15,
                 w_spectral=0.3, w_vgg=0.1, w_adv=5e-3):
        super().__init__()
        self.pixel_loss = CharbonnierLoss()
        self.ssim_loss = SSIMLoss()
        self.spectral_loss = SpectralConsistencyLoss()
        self.vgg_loss = vgg_extractor
        self.adv_criterion = nn.BCEWithLogitsLoss()
        self.w_pixel = w_pixel
        self.w_ssim = w_ssim
        self.w_spectral = w_spectral
        self.w_vgg = w_vgg
        self.w_adv = w_adv

    def forward(self, pred, target, disc_pred_fake=None):
        """Returns (total, parts_dict) so each term can be logged separately."""
        parts = {
            "pixel": self.w_pixel * self.pixel_loss(pred, target),
            "ssim": self.w_ssim * self.ssim_loss(pred, target),
            "spectral": self.w_spectral * self.spectral_loss(pred, target),
        }
        if self.vgg_loss is not None:
            parts["vgg"] = self.w_vgg * self.vgg_loss(pred, target)
        if disc_pred_fake is not None:
            # non-saturating GAN loss: generator wants D to call its output real
            target_real = torch.ones_like(disc_pred_fake)
            parts["adv"] = self.w_adv * self.adv_criterion(disc_pred_fake, target_real)

        total = sum(parts.values())
        return total, {k: v.item() for k, v in parts.items()}


class DiscriminatorLoss(nn.Module):
    """
    Standard BCE real/fake with one-sided label smoothing.

    real_label 0.9 rather than 1.0 keeps the discriminator from becoming
    over-confident, which is the usual way these collapse: a perfect D gives
    the generator no usable gradient.
    """

    def __init__(self, real_label=0.9, fake_label=0.0):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        self.real_label = real_label
        self.fake_label = fake_label

    def forward(self, pred_real, pred_fake):
        loss_real = self.criterion(pred_real, torch.full_like(pred_real, self.real_label))
        loss_fake = self.criterion(pred_fake, torch.full_like(pred_fake, self.fake_label))
        return 0.5 * (loss_real + loss_fake)
