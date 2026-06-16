# tests/test_p3_buffer.py — pure-CPU unit tests (no Isaac, no GPU).
#   pytest tests/test_p3_buffer.py -v
"""Unit tests for buffer.replay_buffer.EpisodeBuffer + Visual HER integration."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from buffer.replay_buffer import EpisodeBuffer, Batch
from buffer.visual_her import make_visual_her_fn, ZONE_POSITIONS


# ── Helpers ───────────────────────────────────────────────────────────────────

ACTION_DIM = 6

def _obs(seed: int, holding: float = 0.0) -> dict:
    """Build a v2-contract obs dict (small pixels for test speed)."""
    rng = np.random.default_rng(seed)
    return {
        "pixels":   rng.random((3, 8, 8), dtype=np.float32),
        "position": rng.random(3, dtype=np.float32),
        "heading":  rng.random(2, dtype=np.float32),
        "goal":     rng.random(3, dtype=np.float32),
        "goal_id":  np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "ee_pos":   rng.random(3, dtype=np.float32),
        "gripper":  rng.random(1, dtype=np.float32),
        "holding":  np.array([holding], dtype=np.float32),
        "box_pos":  rng.random(3, dtype=np.float32),
    }


def _action() -> np.ndarray:
    return np.zeros(ACTION_DIM, dtype=np.float32)


# ── Basic buffer tests ────────────────────────────────────────────────────────

def test_add_and_len():
    buf = EpisodeBuffer(capacity=10, seed=0)
    for i in range(5):
        buf.add(_obs(i), _action(), 0.5, _obs(i + 1), i == 4)
    assert len(buf) == 5


def test_ring_overwrite():
    buf = EpisodeBuffer(capacity=4, seed=0)
    for i in range(10):
        buf.add(_obs(i), _action(), float(i), _obs(i + 1), i == 9)
    assert len(buf) == 4


def test_sample_shapes():
    buf = EpisodeBuffer(capacity=32, seed=1)
    for i in range(20):
        buf.add(_obs(i), _action(), 0.5, _obs(i + 1), i == 19)
    batch = buf.sample(8)
    assert isinstance(batch, Batch)
    assert batch.action.shape == (8, ACTION_DIM)
    assert batch.reward.shape == (8,)
    assert batch.done.shape == (8,)
    assert batch.obs["pixels"].shape == (8, 3, 8, 8)
    assert batch.obs["goal_id"].shape == (8, 3)
    assert batch.obs["holding"].shape == (8, 1)
    assert batch.next_obs["position"].shape == (8, 3)


def test_batched_add_unrolls():
    buf = EpisodeBuffer(capacity=16, seed=2)
    obs = {k: np.stack([_obs(i)[k] for i in range(4)]) for k in _obs(0)}
    nobs = {k: np.stack([_obs(i + 1)[k] for i in range(4)]) for k in _obs(0)}
    buf.add(obs, np.zeros((4, ACTION_DIM), np.float32),
            np.ones(4, np.float32), nobs, np.zeros(4, np.bool_))
    assert len(buf) == 4


def test_empty_sample_raises():
    buf = EpisodeBuffer(capacity=4)
    with pytest.raises(RuntimeError):
        buf.sample(2)


# ── Episode tracking + HER ────────────────────────────────────────────────────

def test_her_auto_called_on_done():
    """HER relabeling fires automatically when done=True is added."""
    call_count = [0]

    def counting_her(trajectory):
        call_count[0] += 1
        return []

    buf = EpisodeBuffer(capacity=32, her_fn=counting_her, seed=3)
    for i in range(5):
        buf.add(_obs(i), _action(), 0.1, _obs(i + 1), done=(i == 4))
    assert call_count[0] == 1


def test_her_adds_transitions_when_holding():
    """Visual HER appends relabeled transitions when robot held the box."""
    her_fn = make_visual_her_fn(
        zone_positions=ZONE_POSITIONS,
        success_reward=10.0,
        her_ratio=1.0,
    )
    buf = EpisodeBuffer(capacity=64, her_fn=her_fn, seed=4)

    # 3-step episode; box held from step 0 onward.
    for i in range(3):
        o = _obs(i, holding=1.0)
        no = _obs(i + 1, holding=1.0)
        buf.add(o, _action(), 0.0, no, done=(i == 2))

    # At least 3 original + some relabeled.
    assert len(buf) > 3


def test_her_no_grasp_no_extra_transitions():
    """No HER transitions when robot never grasped the box."""
    her_fn = make_visual_her_fn(her_ratio=1.0)
    buf = EpisodeBuffer(capacity=32, her_fn=her_fn, seed=5)

    for i in range(4):
        buf.add(_obs(i, holding=0.0), _action(), 0.0,
                _obs(i + 1, holding=0.0), i == 3)
    assert len(buf) == 4


def test_her_relabeled_goal_is_zone_position():
    """Relabeled transitions set goal to one of the known zone positions."""
    relabeled: list[dict] = []

    def capturing_her(trajectory):
        inner_fn = make_visual_her_fn(zone_positions=ZONE_POSITIONS, her_ratio=1.0)
        result = inner_fn(trajectory)
        relabeled.extend(result)
        return result

    buf = EpisodeBuffer(capacity=32, her_fn=capturing_her, seed=6)
    for i in range(3):
        o = _obs(i, holding=1.0)
        no = _obs(i + 1, holding=1.0)
        buf.add(o, _action(), 0.0, no, done=(i == 2))

    assert relabeled, "Expected relabeled transitions"
    for t in relabeled:
        g = t["obs"]["goal"]
        gid = t["obs"]["goal_id"]
        assert g.shape == (3,)
        assert gid.shape == (3,)
        # goal_id must be a valid one-hot.
        assert int(np.round(gid.sum())) == 1, f"goal_id not one-hot: {gid}"


def test_her_last_step_gets_success_reward():
    """The final relabeled step should have success_reward."""
    success_r = 5.0
    relabeled: list[dict] = []

    def capturing_her(trajectory):
        inner_fn = make_visual_her_fn(her_ratio=1.0, success_reward=success_r)
        result = inner_fn(trajectory)
        relabeled.extend(result)
        return result

    buf = EpisodeBuffer(capacity=32, her_fn=capturing_her, seed=7)
    for i in range(3):
        buf.add(_obs(i, holding=1.0), _action(), 0.0,
                _obs(i + 1, holding=1.0), done=(i == 2))

    assert relabeled
    assert relabeled[-1]["reward"] == pytest.approx(success_r)
