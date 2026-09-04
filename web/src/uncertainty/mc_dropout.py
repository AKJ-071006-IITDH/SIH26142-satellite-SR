import torch
import numpy as np

def enable_dropout(model: torch.nn.Module):
    """
    Selectively re-enables Dropout layers during evaluation mode
    for Monte Carlo Dropout uncertainty estimation.
    """
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d)):
            module.train()

def mc_dropout_predict(model: torch.nn.Module, lr_input: torch.Tensor, n_samples: int = 12):
    """
    Executes n_samples stochastic forward passes with dropout active.
    Returns:
      mean_pred: (B, C, H, W) numpy array — mean prediction
      uncertainty: (B, C, H, W) numpy array — pixel-wise standard deviation
    """
    model.eval()
    enable_dropout(model)

    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(lr_input)
            predictions.append(pred.cpu().numpy())

    predictions = np.stack(predictions, axis=0)  # (N, B, C, H, W)
    mean_pred = predictions.mean(axis=0)
    uncertainty = predictions.std(axis=0)

    return mean_pred, uncertainty

def uncertainty_to_confidence_map(uncertainty: np.ndarray):
    """
    Collapses per-band uncertainty (B, C, H, W) into a single 2D spatial confidence heatmap (B, H, W)
    normalized to [0, 1].
    """
    combined = uncertainty.mean(axis=1)  # Average over spectral channels -> (B, H, W)
    normalized = np.zeros_like(combined)
    for i in range(combined.shape[0]):
        img = combined[i]
        min_v, max_v = img.min(), img.max()
        if max_v - min_v > 1e-8:
            normalized[i] = (img - min_v) / (max_v - min_v)
        else:
            normalized[i] = np.zeros_like(img)
    return normalized
