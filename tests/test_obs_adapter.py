# tests/test_obs_adapter.py — pure-CPU unit tests (no Isaac, no GPU).
#   pytest tests/test_obs_adapter.py -v
"""Unit tests for models.dreamerv3.obs_adapter conversions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.dreamerv3.obs_adapter import pixels_to_image, warehouse_obs_to_dreamer


def test_pixels_chw_float_to_hwc_uint8():
    px = np.zeros((1, 3, 64, 64), np.float32)
    px[0, 0] = 1.0  # full red
    img = pixels_to_image(px)
    assert img.shape == (64, 64, 3)
    assert img.dtype == np.uint8
    assert img[0, 0, 0] == 255 and img[0, 0, 1] == 0


def test_pixels_clips_out_of_range():
    px = np.full((3, 4, 4), 2.0, np.float32)  # >1 should clip to 255
    img = pixels_to_image(px)
    assert img.max() == 255


def test_warehouse_obs_to_dreamer_keys():
    obs = {
        "pixels":   np.random.rand(1, 3, 64, 64).astype(np.float32),
        "position": np.random.rand(1, 3).astype(np.float32),
        "heading":  np.random.rand(1, 2).astype(np.float32),
        "goal":     np.random.rand(1, 3).astype(np.float32),
        "goal_id":  np.array([[1.0, 0.0, 0.0]], np.float32),
        "ee_pos":   np.random.rand(1, 3).astype(np.float32),
        "gripper":  np.random.rand(1, 1).astype(np.float32),
        "holding":  np.array([[1.0]], np.float32),
        "box_pos":  np.random.rand(1, 3).astype(np.float32),
    }
    out = warehouse_obs_to_dreamer(obs)
    assert out["image"].shape == (64, 64, 3) and out["image"].dtype == np.uint8
    assert out["position"].shape == (3,)
    assert out["goal_id"].shape == (3,)
    assert out["gripper"].shape == (1,)
    assert out["box_pos"].shape == (3,)
    assert "goal_emb" not in out
