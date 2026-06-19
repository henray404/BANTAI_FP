# models/dreamerv3/obs_adapter.py
# Person 2 — convert WarehouseGymEnv obs ⇄ NM512 dreamerv3-torch obs.
#
# WarehouseGymEnv obs (interface contract — pickup v2, 2026-06-08):
#   pixels  (1,3,64,64) float[0,1] CHW | position (1,3) | heading (1,2) | goal (1,3)
#   goal_id (1,3) one-hot             | ee_pos (1,3) | gripper (1,1) | holding (1,1)
#   box_pos (1,3)
#
# NM512 expects a per-step dict with:
#   image (64,64,3) uint8 HWC  + flat float vector keys (matched by encoder regex)
#   + is_first / is_terminal bookkeeping (added by the suite env, not here).
#
# Pure functions — no Isaac import → unit-testable on CPU with dummy obs.

"""Conversion between the warehouse obs dict and NM512 dreamer obs dict."""

from __future__ import annotations

import numpy as np

# Vector obs keys forwarded to the RSSM MLP encoder (config mlp_keys regex must match).
VECTOR_KEYS = ("position", "heading", "goal", "goal_id",
               "ee_pos", "gripper", "holding", "box_pos")


def _np(x):
    """torch/np → numpy, squeeze a leading batch dim of 1."""
    try:
        import torch

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except ImportError:
        pass
    x = np.asarray(x)
    if x.ndim >= 1 and x.shape[0] == 1:
        x = x[0]
    return x


def pixels_to_image(pixels) -> np.ndarray:
    """(1,3,64,64) or (3,64,64) float[0,1] CHW → (64,64,3) uint8 HWC."""
    px = _np(pixels)               # (3,64,64)
    if px.dtype == np.uint8:
        img = np.transpose(px, (1, 2, 0))
    else:
        img = np.transpose(px, (1, 2, 0))
        img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.ascontiguousarray(img)


def warehouse_obs_to_dreamer(obs: dict) -> dict:
    """Convert a WarehouseGymEnv obs dict → NM512 obs dict (image + flat vectors).

    is_first/is_last/is_terminal are added by WarehouseDreamerEnv, not here.
    """
    out = {"image": pixels_to_image(obs["pixels"])}
    for k in VECTOR_KEYS:
        out[k] = _np(obs[k]).astype(np.float32)
    return out
