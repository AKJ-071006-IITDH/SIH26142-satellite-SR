import torch
import torch.nn as nn
import torch.nn.functional as F

# src/models/rrdbnet_attn.py — attention-enhanced RRDB generator.
#
# Self-contained: does not import from rrdb.py, so this architecture can be
# read and diffed independently of the original RRDBNet track.


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module: channel attention (what to focus
    on) followed by spatial attention (where to focus). Cheap relative to
    the conv stack it sits on top of, and gives the network a way to
    reweight features per-block instead of treating every channel/location
    as equally useful -- which matters more than raw depth when the
    training set is small.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        channel_attn = self.sigmoid(self.channel_mlp(avg_pool) + self.channel_mlp(max_pool))
        x = x * channel_attn

        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = self.sigmoid(self.spatial_conv(torch.cat([avg_map, max_map], dim=1)))
        return x * spatial_attn


class ResidualDenseBlockAttn(nn.Module):
    def __init__(self, channels=64, growth_channels=32, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, growth_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels + growth_channels, growth_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels + 2 * growth_channels, growth_channels, 3, padding=1)
        self.conv4 = nn.Conv2d(channels + 3 * growth_channels, growth_channels, 3, padding=1)
        self.conv5 = nn.Conv2d(channels + 4 * growth_channels, channels, 3, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # Dropout2d (not plain Dropout) so whole feature channels are zeroed,
        # matching src/models/rrdb.py's approach -- needed for the MC-dropout
        # uncertainty pass in src/uncertainity/mc_dropout.py to stay meaningful.
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.cbam = CBAM(channels)

    def forward(self, x):
        x1 = self.lrelu(self.dropout(self.conv1(x)))
        x2 = self.lrelu(self.dropout(self.conv2(torch.cat([x, x1], dim=1))))
        x3 = self.lrelu(self.dropout(self.conv3(torch.cat([x, x1, x2], dim=1))))
        x4 = self.lrelu(self.dropout(self.conv4(torch.cat([x, x1, x2, x3], dim=1))))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        x5 = self.cbam(x5)
        return x + 0.2 * x5


class RRDBAttn(nn.Module):
    """Residual in Residual Dense Block, attention variant: 3 attn-RDBs chained, outer residual."""
    def __init__(self, channels=64, growth_channels=32, dropout_rate=0.1):
        super().__init__()
        self.rdb1 = ResidualDenseBlockAttn(channels, growth_channels, dropout_rate)
        self.rdb2 = ResidualDenseBlockAttn(channels, growth_channels, dropout_rate)
        self.rdb3 = ResidualDenseBlockAttn(channels, growth_channels, dropout_rate)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + 0.2 * out


class UpsampleBlock(nn.Module):
    """Pixel-shuffle upsampling -- same choice as rrdb.py, avoids checkerboard artifacts."""
    def __init__(self, channels=64, scale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * (scale_factor ** 2), 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.lrelu(self.pixel_shuffle(self.conv(x)))


class AttentionRRDBNet(nn.Module):
    """
    Full generator: shallow feature extraction -> stack of attention-RRDBs ->
    feature-space global residual -> upsampling -> output conv -> image-space
    global residual against a bicubic-upsampled copy of the input.

    That last skip is the key fidelity change over rrdb.py's RRDBNet: the
    network only has to learn the residual *detail* on top of a simple
    upsample instead of reconstructing the whole image from scratch, which
    is a much easier target to fit well from ~1,200 training patches.

    num_blocks defaults to 8 (lower than Phase 2's 12) since CBAM attention
    adds representational capacity per block -- depth alone isn't the lever
    here.
    """
    def __init__(self, in_channels=4, out_channels=4, base_channels=64,
                 num_blocks=8, growth_channels=32, scale_factor=4, dropout_rate=0.1):
        super().__init__()
        self.scale_factor = scale_factor

        self.conv_first = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.rrdb_blocks = nn.Sequential(
            *[RRDBAttn(base_channels, growth_channels, dropout_rate) for _ in range(num_blocks)]
        )
        self.conv_after_body = nn.Conv2d(base_channels, base_channels, 3, padding=1)

        assert scale_factor in (2, 4, 8), "build via chained x2 stages"
        num_upsample_stages = {2: 1, 4: 2, 8: 3}[scale_factor]
        self.upsample = nn.Sequential(
            *[UpsampleBlock(base_channels, scale_factor=2) for _ in range(num_upsample_stages)]
        )

        self.conv_hr = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.conv_last = nn.Conv2d(base_channels, out_channels, 3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        # Zero-init the output conv so the network starts as *exactly* the
        # bicubic skip and learns the residual outward from there. Without
        # this, a default-initialized conv_last emits a residual with far
        # more variance than the signal itself -- these are reflectance
        # patches with std ~0.08 -- so training starts by fighting its own
        # random output instead of refining a sane baseline.
        nn.init.zeros_(self.conv_last.weight)
        nn.init.zeros_(self.conv_last.bias)

    def forward(self, x):
        feat = self.conv_first(x)
        body_out = self.conv_after_body(self.rrdb_blocks(feat))
        feat = feat + body_out
        feat = self.upsample(feat)
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))

        base = F.interpolate(x, scale_factor=self.scale_factor, mode="bicubic", align_corners=False)
        return out + base
