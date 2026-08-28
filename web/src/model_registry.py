"""
src/model_registry.py -- NEW FILE

Centralizes loading and caching of your three trained checkpoints
(Phase 2, Phase 3, Phase 4), keyed by a model_id the frontend can
switch between. Reads model definitions from configs/config.yaml so
adding a fourth model later is a config change, not a code change.
"""

import os
import torch
import yaml
from src.models.rrdb import RRDBNet

_loaded_models = {}   # cache: model_id -> loaded nn.Module, so switching
                        # models in the UI doesn't reload from disk every
                        # single request -- only the first request per
                        # model pays the disk-load cost


def load_model_registry(config_path="configs/config.yaml"):
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("models", {})


def get_model(model_id: str, device: str, config_path="configs/config.yaml"):
    """
    Returns a ready-to-use, eval-mode model for the given model_id.
    Raises a clear error if the model_id isn't defined in config.yaml
    or its checkpoint file doesn't exist on disk -- fails loudly and
    early rather than silently falling back to a random-init model,
    which would produce garbage output that LOOKS like a real result.
    """
    cache_key = f"{model_id}:{device}"
    if cache_key in _loaded_models:
        return _loaded_models[cache_key]

    registry = load_model_registry(config_path)
    if model_id not in registry:
        raise ValueError(
            f"Unknown model_id '{model_id}'. Available: {list(registry.keys())}"
        )

    entry = registry[model_id]
    ckpt_path = entry["checkpoint"]

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint for model '{model_id}' not found at {ckpt_path}. "
            f"Make sure the .pt file has been copied into the web project's "
            f"checkpoints/ directory."
        )

    ckpt = torch.load(ckpt_path, map_location=device)
    num_blocks = ckpt.get("num_blocks", entry.get("num_blocks", 6))

    model = RRDBNet(
        in_channels=4, out_channels=4,
        num_blocks=num_blocks, scale_factor=4
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()

    _loaded_models[cache_key] = model
    print(f"Loaded model '{model_id}' ({num_blocks} blocks) from {ckpt_path} onto {device}")
    return model


def list_available_models(config_path="configs/config.yaml"):
    """
    Returns metadata for every model defined in config, plus whether its
    checkpoint file actually exists -- lets the frontend grey out a model
    switch option if that .pt file hasn't been placed yet, instead of
    letting the user select it and hit a confusing 500 error.
    """
    registry = load_model_registry(config_path)
    result = []
    for model_id, entry in registry.items():
        result.append({
            "id": model_id,
            "label": entry.get("label", model_id),
            "description": entry.get("description", ""),
            "num_blocks": entry.get("num_blocks"),
            "available": os.path.exists(entry["checkpoint"]),
        })
    return result