from src.models.rrdb import RRDBNet
import torch
model = RRDBNet(num_blocks=6, scale_factor=4)
out = model(torch.randn(2, 4, 32, 32))
print(out.shape)  # expect torch.Size([2, 4, 128, 128])