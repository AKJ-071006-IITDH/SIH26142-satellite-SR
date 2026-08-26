import torch
import torch.nn as nn

# src/models/rrdb.py — modification to ResidualDenseBlock from Module 4

class ResidualDenseBlock(nn.Module):
    def __init__(self, channels=64, growth_channels=32, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, growth_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels + growth_channels, growth_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels + 2*growth_channels, growth_channels, 3, padding=1)
        self.conv4 = nn.Conv2d(channels + 3*growth_channels, growth_channels, 3, padding=1)
        self.conv5 = nn.Conv2d(channels + 4*growth_channels, channels, 3, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # Dropout2d zeros out entire feature CHANNELS, not individual pixels —
        # this is deliberate for conv layers: dropping single random pixels
        # barely affects a spatially-correlated conv feature map (neighboring
        # pixels compensate), but dropping whole channels genuinely removes
        # a learned feature, giving you meaningful uncertainty signal.
        self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        x1 = self.lrelu(self.dropout(self.conv1(x)))
        x2 = self.lrelu(self.dropout(self.conv2(torch.cat([x, x1], dim=1))))
        x3 = self.lrelu(self.dropout(self.conv3(torch.cat([x, x1, x2], dim=1))))
        x4 = self.lrelu(self.dropout(self.conv4(torch.cat([x, x1, x2, x3], dim=1))))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))  # no dropout on final layer —
                                                                    # you don't want to corrupt
                                                                    # the block's direct output
        return x + 0.2 * x5


class RRDB(nn.Module):
    """Residual in Residual Dense Block: 3 RDBs chained, with an outer residual."""
    def __init__(self, channels=64, growth_channels=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(channels, growth_channels)
        self.rdb2 = ResidualDenseBlock(channels, growth_channels)
        self.rdb3 = ResidualDenseBlock(channels, growth_channels)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + 0.2 * out   # same residual-scaling trick, one level up


class UpsampleBlock(nn.Module):
    """
    Pixel-shuffle upsampling, not transposed convolution.
    Transposed conv is known to produce checkerboard artifacts — visible
    grid-pattern noise from uneven kernel overlap. Pixel-shuffle avoids
    this entirely: it rearranges channels into spatial dimensions instead
    of learning an upsampling kernel, so there's no overlap pattern to
    cause artifacts. This matters a lot for you specifically, since a
    checkerboard artifact could easily be mistaken by a judge (or your
    own uncertainty metric) for a real spatial feature.
    """
    def __init__(self, channels=64, scale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * (scale_factor ** 2), 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.lrelu(self.pixel_shuffle(self.conv(x)))


class RRDBNet(nn.Module):
    """
    Full generator: shallow feature extraction -> stack of RRDBs ->
    global residual -> upsampling -> output conv.
    """
    def __init__(self, in_channels=4, out_channels=4, base_channels=64,
                 num_blocks=16, growth_channels=32, scale_factor=4):
        super().__init__()
        # in_channels=4: matches your R,G,B,NIR bands from Module 2.
        # num_blocks=16: ESRGAN's paper uses 23; start smaller for faster
        # hackathon iteration, scale up later only if you have time and
        # your baseline is already working end-to-end.

        self.conv_first = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.rrdb_blocks = nn.Sequential(
            *[RRDB(base_channels, growth_channels) for _ in range(num_blocks)]
        )
        self.conv_after_body = nn.Conv2d(base_channels, base_channels, 3, padding=1)

        # Upsampling: chain of x2 blocks to reach the target scale factor.
        # scale_factor=4 needs two x2 stages (2*2=4). If you change
        # scale_factor in your degrade() function in Module 3, this needs
        # to match — this is a common source of shape-mismatch bugs.
        assert scale_factor in (2, 4, 8), "build via chained x2 stages"
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
        feat = feat + body_out   # global residual: body learns the RESIDUAL detail,
                                  # not the whole image from scratch — much easier to train
        feat = self.upsample(feat)
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out