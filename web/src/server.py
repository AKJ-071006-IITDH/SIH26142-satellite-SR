import os
import io
import base64
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.model_registry import get_model, list_available_models
from src.uncertainty.mc_dropout import mc_dropout_predict, uncertainty_to_confidence_map
from src.eval.metrics import compute_psnr, compute_ssim, compute_sam, compute_ergas
from src.db.history_store import HistoryStore
from src.data.degrade import degrade
from src.losses.losses import compute_ndvi

app = FastAPI(title="SIH26142 - Super Resolution Mapping API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

history_db = HistoryStore()
device = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_MODEL_ID = "phase2"   # your best-measured checkpoint -- sensible default
                                # so a judge who doesn't touch the model switch
                                # still sees your strongest result


def compute_stretch_bounds(arr: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0):
    """Per-channel (low, high) percentile bounds, reused across LR/SR/GT
    so all three panels get the SAME brightness stretch -- otherwise each
    panel auto-stretching independently could make the SR output look
    artificially brighter/darker than GT purely from the visualization
    step, not from actual model quality."""
    return [tuple(np.percentile(arr[..., c], [low_pct, high_pct])) for c in range(arr.shape[-1])]


def stretch_contrast(arr: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0, bounds=None) -> np.ndarray:
    """
    Per-channel percentile contrast stretch for satellite RGB display.

    Satellite reflectance values physically never approach 1.0 (bright
    surfaces top out around 0.3-0.6) -- a naive value*255 mapping makes
    every RGB preview look dark and muddy even though the data itself
    is fine. This rescales each channel's [low_pct, high_pct] percentile
    range to [0, 1] so the *displayed* image uses the full brightness
    range, purely for visualization -- it does NOT touch the underlying
    data used for metrics/inference.

    Pass `bounds` (from compute_stretch_bounds) to reuse fixed bounds
    instead of computing fresh percentiles from this array.
    """
    out = np.zeros_like(arr, dtype=np.float32)
    for c in range(arr.shape[-1]):
        if bounds is not None:
            lo, hi = bounds[c]
        else:
            lo, hi = np.percentile(arr[..., c], [low_pct, high_pct])
        if hi - lo < 1e-6:
            hi = lo + 1e-6
        out[..., c] = np.clip((arr[..., c] - lo) / (hi - lo), 0.0, 1.0)
    return out


def array_to_base64_png(arr: np.ndarray, is_colormap=False, cmap="inferno", stretch=False, stretch_bounds=None) -> str:
    buf = io.BytesIO()
    if is_colormap:
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        ax.imshow(arr, cmap=cmap)
        ax.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    else:
        if stretch or stretch_bounds is not None:
            display_arr = stretch_contrast(arr[:, :, :3], bounds=stretch_bounds)
        else:
            display_arr = arr
        norm = np.clip(display_arr * 255.0, 0, 255).astype(np.uint8)
        if norm.shape[-1] >= 3:
            img = Image.fromarray(norm[:, :, :3], "RGB")
        else:
            img = Image.fromarray(norm, "L")
        img.save(buf, format="PNG")

    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"


@app.get("/api/status")
def system_status():
    return {
        "status": "online",
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }


@app.get("/api/models")
def get_models():
    """
    Lists all models defined in configs/config.yaml, with an `available`
    flag per model based on whether its .pt file actually exists --
    lets the frontend disable switch options for checkpoints you haven't
    copied into place yet, rather than letting a judge pick one and hit
    a confusing error mid-demo.
    """
    return list_available_models()


@app.get("/api/tiles")
def list_sample_tiles():
    from src.data.fetch import download_or_generate_dataset
    download_or_generate_dataset()
    tiles_dir = "data/raw_tiles"
    files = list(os.listdir(tiles_dir))
    samples = []
    for f in files:
        if f.endswith(".npy"):
            name = f.replace(".npy", "")
            arr = np.load(os.path.join(tiles_dir, f))
            preview_b64 = array_to_base64_png(arr[:128, :128, :3], stretch=True)
            samples.append({
                "id": name,
                "name": name.replace("_", " ").title(),
                "shape": list(arr.shape),
                "preview_b64": preview_b64
            })
    return samples


@app.post("/api/predict")
async def run_super_resolution(
    model_id: str = Form(DEFAULT_MODEL_ID),
    tile_id: str = Form(None),
    n_samples: int = Form(12),
    file: UploadFile = File(None)
):
    try:
        m = get_model(model_id, device)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    tile_name = "Custom Upload"
    has_file = file is not None and getattr(file, "filename", None) and file.filename.strip() != ""

    if has_file:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        
        filename = file.filename.lower()
        tile_name = file.filename or "Uploaded Image"
        
        if filename.endswith(".npy"):
            try:
                arr = np.load(io.BytesIO(contents))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid .npy file: {e}")
            
            if arr.ndim == 2:
                arr = np.stack([arr] * 4, axis=-1)
            elif arr.ndim == 3:
                if arr.shape[0] in [3, 4] and arr.shape[2] not in [3, 4]:
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.shape[-1] == 3:
                    nir = np.clip(arr[:, :, 1] * 1.2 + arr[:, :, 0] * 0.3, 0.0, 1.0)
                    arr = np.dstack([arr, nir])
                elif arr.shape[-1] > 4:
                    arr = arr[:, :, :4]
            hr_np = arr.astype(np.float32)
            if hr_np.max() > 1.0:
                if hr_np.max() <= 255.0:
                    hr_np /= 255.0
                else:
                    hr_np /= hr_np.max()
        else:
            try:
                pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Cannot decode image: {e}")
            img_np = np.array(pil_img).astype(np.float32) / 255.0
            # Synthesize pseudo-NIR for RGB upload
            nir = np.clip(img_np[:, :, 1] * 1.2 + img_np[:, :, 0] * 0.3, 0.0, 1.0)
            hr_np = np.dstack([img_np, nir])
    elif tile_id is not None and tile_id.strip() != "":
        tile_path = f"data/raw_tiles/{tile_id}.npy"
        if not os.path.exists(tile_path):
            raise HTTPException(status_code=404, detail="Sample tile not found")
        hr_np = np.load(tile_path).astype(np.float32)
        tile_name = tile_id.replace("_", " ").title()
    else:
        raise HTTPException(status_code=400, detail="Must provide tile_id or file upload")

    # Ensure dimensions are multiples of 4 to prevent downsample/upsample mismatch
    h, w = hr_np.shape[:2]
    crop_h = max(32, (min(h, 128) // 4) * 4)
    crop_w = max(32, (min(w, 128) // 4) * 4)

    if h < crop_h or w < crop_w:
        pil_res = Image.fromarray(np.clip(hr_np[:, :, :3] * 255.0, 0, 255).astype(np.uint8)).resize((crop_w, crop_h), Image.Resampling.BICUBIC)
        img_np = np.array(pil_res).astype(np.float32) / 255.0
        nir = np.clip(img_np[:, :, 1] * 1.2 + img_np[:, :, 0] * 0.3, 0.0, 1.0)
        hr_crop = np.dstack([img_np, nir])
    else:
        hr_crop = hr_np[:crop_h, :crop_w, :4]

    lr_np = degrade(hr_crop, scale_factor=4)
    lr_tensor = torch.from_numpy(lr_np).permute(2, 0, 1).unsqueeze(0).float().to(device)

    mean_pred, uncertainty = mc_dropout_predict(m, lr_tensor, n_samples=n_samples)
    sr_crop = np.clip(mean_pred[0].transpose(1, 2, 0), 0.0, 1.0)
    confidence_map = uncertainty_to_confidence_map(uncertainty)[0]

    sr_tensor = torch.from_numpy(sr_crop).permute(2, 0, 1).unsqueeze(0)
    ndvi_sr = compute_ndvi(sr_tensor)[0, 0].numpy()

    # Shared brightness-stretch bounds (computed once from ground truth)
    # so LR/SR/GT panels are all displayed on the same visual scale.
    gt_stretch_bounds = compute_stretch_bounds(hr_crop[:, :, :3])

    psnr_val = compute_psnr(sr_crop, hr_crop)
    ssim_val = compute_ssim(sr_crop, hr_crop)
    sam_val = compute_sam(sr_crop, hr_crop)
    ergas_val = compute_ergas(sr_crop, hr_crop, scale_factor=4)

    metrics = {
        "psnr": round(psnr_val, 2),
        "ssim": round(ssim_val, 4),
        "sam": round(sam_val, 2),
        "ergas": round(ergas_val, 2)
    }

    params = {
        "model_id": model_id,
        "scale_factor": 4,
        "n_samples": n_samples,
        "patch_size": f"{crop_h}x{crop_w}"
    }

    lr_b64 = array_to_base64_png(lr_np[:, :, :3], stretch_bounds=gt_stretch_bounds)
    sr_b64 = array_to_base64_png(sr_crop[:, :, :3], stretch_bounds=gt_stretch_bounds)
    gt_b64 = array_to_base64_png(hr_crop[:, :, :3], stretch_bounds=gt_stretch_bounds)   # NEW -- ground truth panel
    uncertainty_b64 = array_to_base64_png(confidence_map, is_colormap=True, cmap="inferno")
    ndvi_b64 = array_to_base64_png(ndvi_sr, is_colormap=True, cmap="YlGn")

    entry_id = history_db.add_entry(
        name=tile_name,
        model_id=model_id,
        lr_b64=lr_b64,
        sr_b64=sr_b64,
        gt_b64=gt_b64,
        uncertainty_b64=uncertainty_b64,
        ndvi_b64=ndvi_b64,
        metrics=metrics,
        params=params
    )

    return {
        "id": entry_id,
        "name": tile_name,
        "model_id": model_id,
        "lr_b64": lr_b64,
        "sr_b64": sr_b64,
        "gt_b64": gt_b64,
        "uncertainty_b64": uncertainty_b64,
        "ndvi_b64": ndvi_b64,
        "metrics": metrics,
        "params": params
    }


@app.get("/api/history")
def get_history():
    return history_db.list_entries()


@app.get("/api/history/{entry_id}")
def get_history_item(entry_id: str):
    entry = history_db.get_entry(entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="History entry not found")
    return entry


@app.delete("/api/history/{entry_id}")
def delete_history_item(entry_id: str):
    success = history_db.delete_entry(entry_id)
    return {"success": success}


@app.delete("/api/history")
def clear_all_history():
    history_db.clear_history()
    return {"success": True}


frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)