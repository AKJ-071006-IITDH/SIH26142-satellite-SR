# Quick smoke test — run from repo root
import torch
from src.models.rrdb import RRDBNet
from src.losses.losses import CombinedSRLoss

device = "cuda" if torch.cuda.is_available() else "cpu"
model = RRDBNet(num_blocks=4, scale_factor=4).to(device)  # small num_blocks for a FAST smoke test
criterion = CombinedSRLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

# Fake batch, standing in for real data — just proving the wiring works
lr_fake = torch.rand(4, 4, 32, 32).to(device)
hr_fake = torch.rand(4, 4, 128, 128).to(device)

losses = []
for step in range(20):
    optimizer.zero_grad()
    pred = model(lr_fake)
    loss = criterion(pred, hr_fake)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

print("Loss trend:", losses[0], "->", losses[-1])
assert losses[-1] < losses[0], "Loss isn't decreasing — something's wrong before you burn real compute"
print("Smoke test passed — safe to run on real data")