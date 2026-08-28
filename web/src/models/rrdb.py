import torch
import torch.nn as nn

class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block with dense channel connections, 0.2 residual scaling,
    and 2D Spatial Dropout for Monte Carlo uncertainty estimation.
    """
    def __init__(self, channels=64, growth_channels=32, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, growth_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels + growth_channels, growth_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels + 2 * growth_channels, growth_channels, 3, padding=1)
        self.conv4 = nn.Conv2d(channels + 3 * growth_channels, growth_channels, 3, padding=1)
        self.conv5 = nn.Conv2d(channels + 4 * growth_channels, channels, 3, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        x1 = self.lrelu(self.dropout(self.conv1(x)))
        x2 = self.lrelu(self.dropout(self.conv2(torch.cat([x, x1], dim=1))))
        x3 = self.lrelu(self.dropout(self.conv3(torch.cat([x, x1, x2], dim=1))))
        x4 = self.lrelu(self.dropout(self.conv4(torch.cat([x, x1, x2, x3], dim=1))))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x + 0.2 * x5


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block chaining 3 RDB blocks."""
    def __init__(self, channels=64, growth_channels=32, dropout_rate=0.1):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(channels, growth_channels, dropout_rate)
        self.rdb2 = ResidualDenseBlock(channels, growth_channels, dropout_rate)
        self.rdb3 = ResidualDenseBlock(channels, growth_channels, dropout_rate)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + 0.2 * out


class UpsampleBlock(nn.Module):
    """Pixel-Shuffle upsampling block avoiding checkerboard artifacts."""
    def __init__(self, channels=64, scale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * (scale_factor ** 2), 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.lrelu(self.pixel_shuffle(self.conv(x)))


class RRDBNet(nn.Module):
    """
    RRDBNet Generator for Satellite Super-Resolution Mapping.
    Optimized for multi-spectral input/output (4 bands: RGB + NIR).
    """
    def __init__(self, in_channels=4, out_channels=4, base_channels=64,
                 num_blocks=8, growth_channels=32, scale_factor=4, dropout_rate=0.1):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv_first = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.rrdb_blocks = nn.Sequential(
            *[RRDB(base_channels, growth_channels, dropout_rate) for _ in range(num_blocks)]
        )
        self.conv_after_body = nn.Conv2d(base_channels, base_channels, 3, padding=1)

        # Upsampling stages (e.g. 4x = two 2x PixelShuffle blocks)
        assert scale_factor in (2, 4, 8), "Scale factor must be 2, 4, or 8"
        num_upsample_stages = {2: 1, 4: 2, 8: 3}[scale_factor]
        self.upsample = nn.Sequential(
            *[UpsampleBlock(base_channels, scale_factor=2) for _ in range(num_upsample_stages)]
        )

        self.conv_hr = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.conv_last = nn.Conv2d(base_channels, out_channels, 3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_out = self.conv_after_body(self.rrdb_blocks(feat))
        feat = feat + body_out  # Global residual connection
        feat = self.upsample(feat)
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out
