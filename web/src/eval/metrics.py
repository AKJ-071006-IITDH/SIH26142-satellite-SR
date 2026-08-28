import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def compute_psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """Computes Peak Signal-to-Noise Ratio (PSNR) in dB."""
    p = np.clip(pred, 0.0, data_range)
    t = np.clip(target, 0.0, data_range)
    return float(peak_signal_noise_ratio(t, p, data_range=data_range))

def compute_ssim(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """Computes Structural Similarity Index (SSIM)."""
    p = np.clip(pred, 0.0, data_range)
    t = np.clip(target, 0.0, data_range)
    return float(structural_similarity(t, p, data_range=data_range, channel_axis=-1))

def compute_sam(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """
    Computes Spectral Angle Mapper (SAM) in degrees.
    Measures spectral signature divergence across channels.
    """
    pred_flat = pred.reshape(-1, pred.shape[-1])
    target_flat = target.reshape(-1, target.shape[-1])

    dot_product = np.sum(pred_flat * target_flat, axis=1)
    pred_norm = np.linalg.norm(pred_flat, axis=1)
    target_norm = np.linalg.norm(target_flat, axis=1)

    cos_angle = dot_product / (pred_norm * target_norm + eps)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angles = np.arccos(cos_angle)
    return float(np.degrees(np.mean(angles)))

def compute_ergas(pred: np.ndarray, target: np.ndarray, scale_factor: int = 4) -> float:
    """
    Computes Relative Global Dimensional Synthesis Error (ERGAS).
    Scale-normalized remote-sensing global error index.
    """
    num_bands = pred.shape[-1]
    band_errors = []
    for b in range(num_bands):
        pred_b = pred[..., b]
        target_b = target[..., b]
        rmse = np.sqrt(np.mean((pred_b - target_b) ** 2))
        mean_v = np.mean(target_b) + 1e-8
        band_errors.append((rmse / mean_v) ** 2)

    return float(100.0 * (1.0 / scale_factor) * np.sqrt(np.mean(band_errors)))

def evaluate_batch(pred: torch.Tensor, target: torch.Tensor, scale_factor: int = 4) -> dict:
    """
    Takes (B, C, H, W) PyTorch tensors and returns averaged PSNR, SSIM, SAM, and ERGAS.
    """
    pred_np = pred.detach().cpu().permute(0, 2, 3, 1).numpy()
    target_np = target.detach().cpu().permute(0, 2, 3, 1).numpy()

    metrics = {"psnr": [], "ssim": [], "sam": [], "ergas": []}
    for i in range(pred_np.shape[0]):
        p, t = pred_np[i], target_np[i]
        metrics["psnr"].append(compute_psnr(p, t))
        metrics["ssim"].append(compute_ssim(p, t))
        metrics["sam"].append(compute_sam(p, t))
        metrics["ergas"].append(compute_ergas(p, t, scale_factor))

    return {k: float(np.mean(v)) for k, v in metrics.items()}
