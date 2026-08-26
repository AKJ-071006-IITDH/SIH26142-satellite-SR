from src.data.dataset import SatelliteSRDataset
import matplotlib.pyplot as plt

ds = SatelliteSRDataset()
lr, hr = ds[0]
print("LR shape:", lr.shape, "HR shape:", hr.shape)  # should be ~4x apart spatially

fig, axes = plt.subplots(1, 2)
axes[0].imshow(lr[:3].permute(1,2,0).numpy())  # RGB channels only
axes[1].imshow(hr[:3].permute(1,2,0).numpy())
plt.show()