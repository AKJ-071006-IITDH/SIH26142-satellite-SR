# src/models/discriminator.py
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=4, base_features=64):
        super().__init__()
        
        # Spectral Normalization restricts the Lipschitz constant of every layer to 1,
        # preventing gradient explosion and forcing stable adversarial training.
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, base_features, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(base_features, base_features, 3, 2, 1)),
            nn.BatchNorm2d(base_features),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(base_features, base_features * 2, 3, 1, 1)),
            nn.BatchNorm2d(base_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(base_features * 2, base_features * 2, 3, 2, 1)),
            nn.BatchNorm2d(base_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(base_features * 2, base_features * 4, 3, 1, 1)),
            nn.BatchNorm2d(base_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(base_features * 4, 1, 3, 1, 1))
        )

    def forward(self, x):
        return self.model(x)