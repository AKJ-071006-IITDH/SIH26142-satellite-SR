# src/uncertainty/mc_dropout.py
import torch
import numpy as np

def enable_dropout(model):
    """
    Critical function. model.eval() normally turns OFF all dropout
    (as covered in Module 6). We need dropout active while everything
    ELSE (like any batchnorm-equivalent layers, if you have any) stays
    in eval mode. This function walks the model and selectively
    re-enables just the Dropout layers.
    """
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d)):
            module.train()  # only this layer type goes back to train-mode behavior


def mc_dropout_predict(model, lr_input, n_samples=20):
    """
    Runs the SAME input through the model n_samples times with dropout
    active, collecting a distribution of outputs instead of one point
    estimate.

    n_samples=20 is a reasonable hackathon-timeline default: enough for a
    stable variance estimate, not so many that inference becomes painfully
    slow during a live demo. (Research papers often use 50-100; you can
    mention that as a "future work" scaling note.)
    """
    model.eval()          # turn everything to eval mode first...
    enable_dropout(model) # ...then selectively re-enable dropout only

    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(lr_input)
            predictions.append(pred.cpu().numpy())

    predictions = np.stack(predictions, axis=0)  # shape: (n_samples, B, C, H, W)

    mean_pred = predictions.mean(axis=0)          # your actual SR output — average
                                                     # over samples is more stable than
                                                     # any single stochastic pass
    uncertainty = predictions.std(axis=0)          # per-pixel, per-band standard
                                                     # deviation — THIS is your
                                                     # uncertainty map

    return mean_pred, uncertainty


def uncertainty_to_confidence_map(uncertainty, method="mean_over_bands"):
    """
    uncertainty: (B, C, H, W) numpy array from mc_dropout_predict

    Collapses per-band uncertainty into a single (H, W) heatmap per image
    that you can directly overlay on the output — the actual demo-facing
    artifact.
    """
    # Average uncertainty across spectral bands, per pixel
    combined = uncertainty.mean(axis=1)  # (B, H, W)

    # Normalize to [0, 1] for visualization — min-max per image so the
    # heatmap uses the full color range regardless of the raw scale
    normalized = np.zeros_like(combined)
    for i in range(combined.shape[0]):
        img = combined[i]
        normalized[i] = (img - img.min()) / (img.max() - img.min() + 1e-8)

    return normalized  # 0 = most confident, 1 = least confident/most inferred