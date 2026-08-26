import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def compute_psnr(pred: np.ndarray, target: np.ndarray, data_range=1.0):
    """
    pred, target: (H, W, C) numpy arrays, same value range (0-1 assumed here
    — make sure your data is actually normalized to this range before
    calling, otherwise the number is meaningless).
    """
    return peak_signal_noise_ratio(target, pred, data_range=data_range)


def compute_ssim(pred: np.ndarray, target: np.ndarray, data_range=1.0):
    """
    channel_axis=-1 tells skimage your array is (H, W, C) not (H, W) —
    easy to get wrong and silently compute SSIM on only one band if
    you forget this argument.
    """
    return structural_similarity(target, pred, data_range=data_range, channel_axis=-1)


def compute_sam(pred: np.ndarray, target: np.ndarray, eps=1e-8):
    """
    pred, target: (H, W, C) — C is your spectral bands (4, in your case)
    Returns the mean spectral angle across all pixels, in degrees.

    The math: for each pixel, treat its band values as a vector. The angle
    between the predicted vector and true vector is:
        angle = arccos( (pred . target) / (|pred| * |target|) )
    A perfect spectral match gives angle = 0. Larger angle = the model
    is distorting the relationship between bands, even if overall
    brightness (magnitude) happens to be close.
    """
    pred_flat = pred.reshape(-1, pred.shape[-1])
    target_flat = target.reshape(-1, target.shape[-1])

    dot_product = np.sum(pred_flat * target_flat, axis=1)
    pred_norm = np.linalg.norm(pred_flat, axis=1)
    target_norm = np.linalg.norm(target_flat, axis=1)

    cos_angle = dot_product / (pred_norm * target_norm + eps)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # guard against floating-point
                                                 # rounding pushing slightly past
                                                 # [-1, 1], which would make
                                                 # arccos return NaN
    angles = np.arccos(cos_angle)
    return np.degrees(np.mean(angles))


def compute_ergas(pred: np.ndarray, target: np.ndarray, scale_factor: int):
    """
    ERGAS formula:
        ERGAS = 100 * (1/scale_factor) * sqrt( mean_over_bands( (RMSE_band / mean_band)^2 ) )

    scale_factor: the resolution ratio between LR input and HR output
    (should match whatever you used in Module 3's degrade() function —
    another spot where a mismatched constant between modules causes
    silently wrong numbers rather than a crash, so double check this).
    """
    num_bands = pred.shape[-1]
    band_errors = []
    for b in range(num_bands):
        pred_band = pred[..., b]
        target_band = target[..., b]
        rmse = np.sqrt(np.mean((pred_band - target_band) ** 2))
        mean_val = np.mean(target_band) + 1e-8
        band_errors.append((rmse / mean_val) ** 2)

    return 100 * (1.0 / scale_factor) * np.sqrt(np.mean(band_errors))


def evaluate_batch(pred: torch.Tensor, target: torch.Tensor, scale_factor: int = 4):
    """
    Convenience wrapper: takes a batch of model outputs (B, C, H, W) tensors,
    returns averaged metrics across the batch as plain floats — this is
    what you'll actually call from your training/eval scripts.
    """
    pred_np = pred.detach().cpu().permute(0, 2, 3, 1).numpy()   # -> (B, H, W, C)
    target_np = target.detach().cpu().permute(0, 2, 3, 1).numpy()

    metrics = {"psnr": [], "ssim": [], "sam": [], "ergas": []}
    for i in range(pred_np.shape[0]):
        p, t = np.clip(pred_np[i], 0, 1), np.clip(target_np[i], 0, 1)
        metrics["psnr"].append(compute_psnr(p, t))
        metrics["ssim"].append(compute_ssim(p, t))
        metrics["sam"].append(compute_sam(p, t))
        metrics["ergas"].append(compute_ergas(p, t, scale_factor))

    return {k: float(np.mean(v)) for k, v in metrics.items()}