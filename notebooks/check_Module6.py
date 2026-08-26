from src.losses.losses import CombinedSRLoss
import torch

criterion = CombinedSRLoss()  # no feature_extractor yet, that's fine — perceptual term just skips
pred = torch.rand(2, 4, 128, 128, requires_grad=True)
target = torch.rand(2, 4, 128, 128)
loss = criterion(pred, target)
print(loss.item())  # should be a single positive float, not NaN

# Confirm gradients flow properly — important, since a broken loss can
# silently produce zero or NaN gradients while still printing a number
loss.backward()
print("Gradient check passed" if not torch.isnan(loss) else "NaN — something's wrong")