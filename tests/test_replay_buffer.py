# tests/test_replay_buffer.py — pure-CPU unit tests (no Isaac, no GPU).
#   pytest tests/test_replay_buffer.py -v
"""Unit tests for training.replay_buffer.ReplayBuffer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.replay_buffer import ReplayBuffer


def _obs(seed: int) -> dict:
    """Build a contract-shaped obs dict with deterministic values."""
    rng = np.random.default_rng(seed)
    return {
        "pixels":   rng.random((3, 8, 8), dtype=np.float32),
        "position": rng.random(3, dtype=np.float32),
        "goal":     rng.random(3, dtype=np.float32),
        "goal_emb": rng.random(512, dtype=np.float32),
        "heading":  rng.random(2, dtype=np.float32),
    }


def test_add_and_len():
    buf = ReplayBuffer(capacity=10, seed=0)
    for i in range(5):
        buf.add(_obs(i), np.array([0.1, -0.2], np.float32), 1.0, _obs(i + 1), False)
    assert len(buf) == 5


def test_ring_overwrite():
    buf = ReplayBuffer(capacity=4, seed=0)
    for i in range(10):
        buf.add(_obs(i), np.array([0.0, 0.0], np.float32), float(i), _obs(i + 1), False)
    assert len(buf) == 4  # capped at capacity


def test_sample_shapes():
    buf = ReplayBuffer(capacity=32, seed=1)
    for i in range(20):
        buf.add(_obs(i), np.array([0.1, 0.1], np.float32), 0.5, _obs(i + 1), bool(i % 2))
    batch = buf.sample(8)
    assert batch.action.shape == (8, 2)
    assert batch.reward.shape == (8,)
    assert batch.done.shape == (8,)
    assert batch.obs["pixels"].shape == (8, 3, 8, 8)
    assert batch.obs["goal_emb"].shape == (8, 512)
    assert batch.next_obs["position"].shape == (8, 3)


def test_batched_add_unrolls():
    buf = ReplayBuffer(capacity=16, seed=2)
    # batched: 4 envs at once
    obs = {k: np.stack([_obs(i)[k] for i in range(4)]) for k in _obs(0)}
    nobs = {k: np.stack([_obs(i + 1)[k] for i in range(4)]) for k in _obs(0)}
    buf.add(obs, np.zeros((4, 2), np.float32), np.arange(4, dtype=np.float32),
            nobs, np.zeros(4, np.bool_))
    assert len(buf) == 4


def test_empty_sample_raises():
    buf = ReplayBuffer(capacity=4)
    try:
        buf.sample(2)
        assert False, "expected RuntimeError on empty buffer"
    except RuntimeError:
        pass
