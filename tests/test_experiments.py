# tests/test_experiments.py
# Pure unit tests for the ablation harness — NO Isaac Sim / torch env needed.
#   pytest tests/test_experiments.py -v
#
# Covers: experiments.configs, experiments.analyze (stats), perception.detection.slope
# state potential, and env.her_nm512 episode relabeling.

"""Unit tests for the experiment harness (config registry, stats, HER, CA-SLOPE)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json as _json
import tempfile

from experiments import analyze, configs
from experiments.settings import load_settings
from experiments.trajectory_recorder import TRAJ_HEADER, TrajectoryRecorder
from env.her_nm512 import relabel_cache_episode


# ── configs registry ──────────────────────────────────────────────────────────

def test_six_configs_eighteen_runs():
    assert len(configs.CONFIGS) == 6
    assert len(configs.all_runs()) == 18
    assert {c.idx for c in configs.CONFIGS} == {1, 2, 3, 4, 5, 6}


def test_factorial_flags():
    # The 2x2 DreamerV3 quadrant must cover all (ca_slope, her) combinations.
    quad = {(c.ca_slope, c.visual_her) for c in configs.CONFIGS if c.algo == "dreamer"}
    assert quad == {(False, False), (True, False), (False, True), (True, True)}


def test_isolation_comparisons_reference_real_configs():
    for a, b, _ in configs.ISOLATION_COMPARISONS:
        configs.by_idx(a)
        configs.by_idx(b)


# ── analyze: stats ─────────────────────────────────────────────────────────────

def test_aggregate_mean_std():
    mean, std = analyze.aggregate([0.0, 0.5, 1.0])
    assert mean == 0.5
    assert abs(std - np.std([0.0, 0.5, 1.0])) < 1e-9


def test_steps_to_threshold():
    rows = [{"step": 10_000, "success_rate": 0.2},
            {"step": 20_000, "success_rate": 0.6},
            {"step": 30_000, "success_rate": 0.9}]
    assert analyze.steps_to_threshold(rows, 0.5) == 20_000
    assert analyze.steps_to_threshold(rows, 1.0) is None


def test_final_success():
    rows = [{"step": 10_000, "success_rate": 0.2}, {"step": 20_000, "success_rate": 0.7}]
    assert analyze.final_success(rows) == 0.7


def test_mann_whitney_separated():
    # Perfectly separated 3v3 -> max U=9; both scipy (asymptotic ~0.05) and the exact
    # fallback (0.1) flag it as the most-significant result the test can produce.
    u, p, method = analyze.mann_whitney_u([1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    assert u == 9.0
    assert p <= 0.1 + 1e-9
    assert method in ("scipy", "exact-perm")


def test_exact_perm_p_is_one_tenth():
    # Exact permutation two-sided p for a perfectly separated 3v3 split is 2/20 = 0.1.
    assert abs(analyze._exact_two_sided_p([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]) - 0.1) < 1e-9


def test_mann_whitney_identical():
    _, p, _ = analyze.mann_whitney_u([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    assert p == 1.0


# ── settings (YAML config loader) ──────────────────────────────────────────────

def test_settings_defaults_match_spec():
    s = load_settings()
    assert s.budget["total_steps"] == 200_000
    assert s.budget["seeds"] == [0, 1, 2]
    assert s.ca_slope["mode"] == "category"
    assert s.ca_slope["category_gains"] == [1.0, 1.5, 2.0]


def test_settings_override_is_deep_merged():
    s = load_settings(overrides={"ca_slope": {"mode": "generic"},
                                 "budget": {"total_steps": 5000}})
    assert s.ca_slope["mode"] == "generic"
    assert s.ca_slope["category_gains"] == [1.0, 1.5, 2.0]  # untouched keys survive
    assert s.budget["total_steps"] == 5000
    assert s.budget["seeds"] == [0, 1, 2]                   # untouched keys survive


# ── TrajectoryRecorder (best-episode action trace for GUI replay) ───────────────

def test_trajectory_recorder_promotes_best():
    import csv
    d = tempfile.mkdtemp()
    rec = TrajectoryRecorder(d)

    rec.begin({"goal_id": [1, 0, 0]})
    rec.step(1, [0.1] * 6, [0, 0, 0], [0, 0, 0], 0.0, -1.0)
    assert rec.end(0.0, -5.0) is True            # first non-empty episode -> saved

    rec.begin({"goal_id": [1, 0, 0]})
    rec.step(1, [0.2] * 6, [1, 1, 1], [1, 1, 1], 0.0, 0.0)
    assert rec.end(0.0, -9.0) is False           # lower return, same success -> not saved

    rec.begin({"goal_id": [0, 1, 0]})
    rec.step(1, [0.3, 0, 0, 0, 0, 1], [2, 2, 2], [1, 1, 1], 1.0, 10.0)
    assert rec.end(1.0, 12.0) is True            # higher success -> saved

    rows = list(csv.reader((Path(d) / "best_trajectory.csv").open()))
    assert tuple(rows[0]) == TRAJ_HEADER
    assert len(rows) == 2                          # header + the 1 step of the best episode
    meta = _json.loads((Path(d) / "best_init.json").read_text())
    assert meta["success_rate"] == 1.0
    assert meta["init"]["goal_id"] == [0, 1, 0]


def test_trajectory_recorder_skips_empty_episode():
    rec = TrajectoryRecorder(tempfile.mkdtemp())
    rec.begin({})
    assert rec.end(1.0, 100.0) is False            # no steps recorded -> nothing saved


# CA-SLOPE is tested in tests/test_ca_slope.py (teammate's canonical reward/ca_slope.py).


# ── Visual HER relabeling (NM512 cache format) ─────────────────────────────────

def _toy_episode(grasped: bool) -> dict:
    """Build a 4-step NM512 cache episode; robot ends near zone index 1 (cyan)."""
    T = 4
    holding = np.array([0, 0, 1, 1] if grasped else [0, 0, 0, 0], np.float32)[:, None]
    # zone 1 (cyan) is at (0, -12); place the robot there after grasp.
    position = np.array([[0, 10, 0], [0, 0, 0], [0, -11, 0], [0, -12, 0]], np.float32)
    return {
        "holding": holding,
        "position": position,
        "goal": np.zeros((T, 3), np.float32),
        "goal_id": np.tile([1, 0, 0], (T, 1)).astype(np.float32),  # commanded orange
        "reward": np.zeros(T, np.float32),
        "is_terminal": np.zeros(T, np.float32),
        "discount": np.ones(T, np.float32),
    }


def test_her_skips_when_never_grasped():
    assert relabel_cache_episode(_toy_episode(grasped=False)) is None


def test_her_relabels_to_achieved_zone():
    out = relabel_cache_episode(_toy_episode(grasped=True), success_reward=10.0)
    assert out is not None
    # Relabeled to the achieved zone (index 1 = cyan at (0,-12)).
    np.testing.assert_allclose(out["goal_id"][-1], [0, 1, 0])
    np.testing.assert_allclose(out["goal"][-1], [0.0, -12.0, 0.01])
    assert out["reward"][-1] == 10.0
    assert bool(out["is_terminal"][-1]) is True
    assert out["discount"][-1] == 0.0
