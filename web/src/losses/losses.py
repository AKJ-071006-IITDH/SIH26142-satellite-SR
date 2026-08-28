import torch
import torch.nn as nn
import torch.nn.functional as F

class CharbonnierLoss(nn.Module):
    """Smooth L1 Loss variant: sqrt((pred - target)^2 + eps^2)"""
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


def compute_ndvi(img, red_idx=0, nir_idx=3, eps=1e-6):
    """
    Computes Normalized Difference Vegetation Index (NDVI) map.
    img: (B, C, H, W) tensor where bands are [Red, Green, Blue, NIR]
    NDVI = (NIR - Red) / (NIR + Red + eps)
    """
    red = img[:, red_idx:red_idx+1, :, :]
    nir = img[:, nir_idx:nir_idx+1, :, :]
    return (nir - red) / (nir + red + eps)


class SpectralConsistencyLoss(nn.Module):
    """
    Remote-Sensing Spectral Loss penalizing deviations in vegetation indices (NDVI)
    and band ratio relationships.
    """
    def forward(self, pred, target):
        ndvi_pred = compute_ndvi(pred)
        ndvi_target = compute_ndvi(target)
        return F.l1_loss(ndvi_pred, ndvi_target)


class SpectralPerceptualLoss(nn.Module):
    """
    Self-feature perceptual loss comparing intermediate feature representations
    to preserve structural detail without ImageNet VGG domain mismatch.
    """
    def __init__(self, feature_extractor: nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.feature_extractor.eval()

    def forward(self, pred, target):
        pred_feat = self.feature_extractor(pred)
        target_feat = self.feature_extractor(target)
        return F.l1_loss(pred_feat, target_feat)


class CombinedSRLoss(nn.Module):
    """
    Combined Remote-Sensing Loss:
    L = w_pixel * L_pixel + w_spectral * L_spectral (+ w_perceptual * L_perceptual)
    """
    def __init__(self, feature_extractor=None, w_pixel=1.0, w_spectral=0.5, w_perceptual=0.1):
        super().__init__()
        self.pixel_loss = CharbonnierLoss()
        self.spectral_loss = SpectralConsistencyLoss()
        self.perceptual_loss = SpectralPerceptualLoss(feature_extractor) if feature_extractor else None
        self.w_pixel = w_pixel
        self.w_spectral = w_spectral
        self.w_perceptual = w_perceptual

    def forward(self, pred, target):
        loss = self.w_pixel * self.pixel_loss(pred, target)
        loss = loss + self.w_spectral * self.spectral_loss(pred, target)
        if self.perceptual_loss is not None:
            loss = loss + self.w_perceptual * self.perceptual_loss(pred, target)
        return loss
