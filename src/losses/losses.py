import torch
import torch.nn as nn
import torch.nn.functional as F

class CharbonnierLoss(nn.Module):
    """
    Smooth L1 variant: sqrt((pred - target)^2 + eps^2).
    eps is tiny (1e-3) — just enough to keep the gradient well-behaved
    near zero difference, without meaningfully changing the loss value
    anywhere else.
    """
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))

class SpectralPerceptualLoss(nn.Module):
    """
    Instead of ImageNet-VGG features, use a small CNN trained (even briefly)
    on your own Sentinel-2 patches for a pretext task like reconstruction
    or band prediction, and compare its intermediate features instead.

    For hackathon timeline: simplest viable version is to reuse your OWN
    RRDBNet's early feature-extraction layers (before the RRDB blocks) as
    a fixed feature extractor, since those layers already learn satellite-
    relevant low-level features once your model is partially trained.
    This is a legitimate shortcut — you're not claiming a separately
    pretrained foundation model, just using self-similarity of early
    features, which is honest to describe in your presentation.
    """
    def __init__(self, feature_extractor: nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor
        for p in self.feature_extractor.parameters():
            p.requires_grad = False   # frozen — we don't want the loss's own
                                        # feature extractor being trained by
                                        # the loss it's computing, that's circular
        self.feature_extractor.eval()

    def forward(self, pred, target):
        pred_feat = self.feature_extractor(pred)
        target_feat = self.feature_extractor(target)
        return F.l1_loss(pred_feat, target_feat)


def compute_ndvi(img, red_idx=0, nir_idx=3, eps=1e-2):
    """
    img: (B, C, H, W) tensor, bands ordered [R, G, B, NIR] per Module 2's evalscript
    NDVI = (NIR - Red) / (NIR + Red) — standard vegetation index formula,
    values range roughly [-1, 1], healthy vegetation is typically 0.2-0.8
    """
    red = img[:, red_idx:red_idx+1, :, :]
    nir = img[:, nir_idx:nir_idx+1, :, :]
    return (nir - red) / (nir + red + eps)


class SpectralConsistencyLoss(nn.Module):
    """
    Penalizes the model if the super-resolved output's NDVI differs from
    the ground truth's NDVI more than pixel differences alone would predict.
    This is what makes your loss function 'remote-sensing aware' rather
    than a generic photo-SR loss with satellite data plugged in.
    """
    def forward(self, pred, target):
        ndvi_pred = compute_ndvi(pred)
        ndvi_target = compute_ndvi(target)
        return F.l1_loss(ndvi_pred, ndvi_target)




class CombinedSRLoss(nn.Module):
    def __init__(self, feature_extractor=None,
                 w_pixel=1.0, w_perceptual=0.1, w_spectral=0.5):
        """
        Weights matter a lot here, and these starting values are
        deliberately pixel-loss-dominant: pixel loss is your most stable,
        reliable signal, and perceptual/spectral losses are refinements
        on top. If you set w_perceptual or w_spectral too high early in
        training, the model can chase 'plausible texture' or 'right NDVI
        on average' at the expense of actually being close to the ground
        truth image — start conservative, increase gradually if outputs
        look pixel-accurate but perceptually flat.
        """
        super().__init__()
        self.pixel_loss = CharbonnierLoss()
        self.spectral_loss = SpectralConsistencyLoss()
        self.perceptual_loss = SpectralPerceptualLoss(feature_extractor) if feature_extractor else None
        self.w_pixel = w_pixel
        self.w_perceptual = w_perceptual
        self.w_spectral = w_spectral

    def forward(self, pred, target):
        loss = self.w_pixel * self.pixel_loss(pred, target)
        loss = loss + self.w_spectral * self.spectral_loss(pred, target)
        if self.perceptual_loss is not None:
            loss = loss + self.w_perceptual * self.perceptual_loss(pred, target)
        return loss


