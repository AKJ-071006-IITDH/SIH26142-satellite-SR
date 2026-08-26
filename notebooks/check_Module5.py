from src.models.rrdb import RRDBNet
import torch

model = RRDBNet(in_channels=4, out_channels=4, num_blocks=16, scale_factor=4)
lr_input = torch.randn(2, 4, 32, 32)   # batch of 2, 4 bands, 32x32 LR patch
hr_output = model(lr_input)
print(hr_output.shape)   # should be torch.Size([2, 4, 128, 128]) — exactly 4x spatial upscale

num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params:,}")   # sanity check it's a reasonable size, not absurdly huge