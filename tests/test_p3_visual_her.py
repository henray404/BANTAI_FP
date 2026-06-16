# tests/test_p3_visual_her.py — pure-CPU unit tests (no Isaac, no GPU).
#   pytest tests/test_p3_visual_her.py -v
"""Unit tests for buffer.visual_her — goal_id-based Visual HER relabeling."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from buffer.visual_her import make_visual_her_fn, ZONE_POSITIONS, ZONE_GOAL_IDS


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_step(
    seed: int,
    holding: float = 0.0,
    position: np.ndarray | None = None,
) -> dict:
    """Build one trajectory step dict (v2 obs contract + action/reward/done)."""
    rng = np.random.default_rng(seed)
    obs = {
        "pixels":   rng.random((3, 8, 8), dtype=np.float32),
        "position": position if position is not None else rng.random(3, dtype=np.float32),
        "heading":  rng.random(2, dtype=np.float32),
        "goal":     ZONE_POSITIONS[0].copy(),
        "goal_id":  np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "ee_pos":   rng.random(3, dtype=np.float32),
        "gripper":  rng.random(1, dtype=np.float32),
        "holding":  np.array([holding], dtype=np.float32),
        "box_pos":  rng.random(3, dtype=np.float32),
    }
    next_obs = {k: v.copy() for k, v in obs.items()}
    return {
        "obs": obs,
        "action": np.zeros(6, dtype=np.float32),
        "reward": 0.0,
        "next_obs": next_obs,
        "done": False,
    }


def _traj(n_steps: int, holding_from: int = 0) -> list[dict]:
    """n-step trajectory; holding≥0.5 from `holding_from` onward."""
    steps = []
    for i in range(n_steps):
        h = 1.0 if i >= holding_from else 0.0
        steps.append(_make_step(seed=i, holding=h))
    steps[-1]["done"] = True
    return steps


# ── Constants ──────────────────────────────────────────────────────────────────

def test_zone_positions_shape():
    assert ZONE_POSITIONS.shape == (3, 3)
    assert ZONE_GOAL_IDS.shape == (3, 3)


def test_zone_goal_ids_are_one_hot():
    for i in range(3):
        assert ZONE_GOAL_IDS[i, i] == 1.0
        assert ZONE_GOAL_IDS[i].sum() == pytest.approx(1.0)


# ── No-grasp cases ─────────────────────────────────────────────────────────────

def test_no_grasp_returns_empty():
    her_fn = make_visual_her_fn(her_ratio=1.0)
    traj = _traj(5, holding_from=999)  # never holding
    result = her_fn(traj)
    assert result == []


def test_empty_trajectory_returns_empty():
    her_fn = make_visual_her_fn(her_ratio=1.0)
    assert her_fn([]) == []


# ── her_ratio gating ───────────────────────────────────────────────────────────

def test_her_ratio_zero_never_relabels():
    her_fn = make_visual_her_fn(her_ratio=0.0)
    traj = _traj(5, holding_from=0)
    results = [her_fn(traj) for _ in range(20)]
    assert all(r == [] for r in results), "her_ratio=0 must never relabel"


def test_her_ratio_one_always_relabels():
    her_fn = make_visual_her_fn(her_ratio=1.0)
    traj = _traj(5, holding_from=0)
    for _ in range(5):
        result = her_fn(traj)
        assert len(result) > 0, "her_ratio=1.0 must always relabel"


# ── Relabeled content correctness ─────────────────────────────────────────────

def test_relabeled_goal_matches_zone_position():
    her_fn = make_visual_her_fn(her_ratio=1.0)
    traj = _traj(5, holding_from=0)
    relabeled = her_fn(traj)
    assert relabeled
    for step in relabeled:
        g = step["obs"]["goal"]
        match = any(np.allclose(g, zp) for zp in ZONE_POSITIONS)
        assert match, f"relabeled goal {g} not a valid zone position"


def test_relabeled_goal_id_is_one_hot():
    her_fn = make_visual_her_fn(her_ratio=1.0)
    traj = _traj(5, holding_from=0)
    relabeled = her_fn(traj)
    for step in relabeled:
        gid = step["obs"]["goal_id"]
        assert gid.shape == (3,)
        assert gid.sum() == pytest.approx(1.0), f"goal_id not one-hot: {gid}"
        assert set(gid.tolist()).issubset({0.0, 1.0})


def test_relabeled_goal_id_consistent_with_goal():
    her_fn = make_visual_her_fn(her_ratio=1.0)
    traj = _traj(5, holding_from=0)
    relabeled = her_fn(traj)
    for step in relabeled:
        g = step["obs"]["goal"]
        gid = step["obs"]["goal_id"]
        zone_idx = int(np.argmax(gid))
        assert np.allclose(ZONE_POSITIONS[zone_idx], g), (
            f"goal_id[{zone_idx}]=1 but goal={g} doesn't match zone {ZONE_POSITIONS[zone_idx]}"
        )


def test_last_step_gets_success_reward():
    success_r = 7.5
    her_fn = make_visual_her_fn(her_ratio=1.0, success_reward=success_r)
    traj = _traj(4, holding_from=0)
    relabeled = her_fn(traj)
    assert relabeled
    assert relabeled[-1]["reward"] == pytest.approx(success_r)


def test_non_last_steps_have_zero_reward():
    her_fn = make_visual_her_fn(her_ratio=1.0, success_reward=10.0)
    traj = _traj(5, holding_from=0)
    relabeled = her_fn(traj)
    assert len(relabeled) >= 2, "Need at least 2 steps to check non-terminal rewards"
    for step in relabeled[:-1]:
        assert step["reward"] == pytest.approx(0.0)


# ── Nearest zone selection ─────────────────────────────────────────────────────

def test_nearest_zone_selected():
    """Robot near zone_B=(0,-12,0.01) → relabeled goal should be zone_B."""
    her_fn = make_visual_her_fn(her_ratio=1.0)

    near_zone_b = np.array([0.1, -11.8, 0.01], dtype=np.float32)
    traj = _traj(4, holding_from=0)
    for step in traj:
        step["obs"]["position"] = near_zone_b.copy()
        step["next_obs"]["position"] = near_zone_b.copy()

    relabeled = her_fn(traj)
    assert relabeled
    g = relabeled[0]["obs"]["goal"]
    assert np.allclose(g, ZONE_POSITIONS[1], atol=1e-3), (
        f"Expected zone_B {ZONE_POSITIONS[1]}, got {g}"
    )


# ── Post-grasp range ───────────────────────────────────────────────────────────

def test_only_post_grasp_steps_relabeled():
    """With holding_from=2, first 2 steps should not be in relabeled list."""
    her_fn = make_visual_her_fn(her_ratio=1.0)
    traj = _traj(6, holding_from=2)
    relabeled = her_fn(traj)
    # Relabeled count should equal number of post-grasp steps (indices 2..5 = 4 steps).
    assert len(relabeled) == 4
