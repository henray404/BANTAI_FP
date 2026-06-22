# tests/test_select_best.py
# Person 5 — per-scenario run scoring + best-run selection. Pure stdlib, runs on a Mac.

import json

from recording.select_best import (
    best_per_scenario, group_by_scenario, load_runs, rank_scenarios, scenario_metrics,
)


def _write_run(tmp_path, name, scenario, seed, success, steps, deliver_step, ret=0.0):
    """Write a minimal <name>.csv + .meta.json the way the recorder would."""
    stem = tmp_path / name
    stem.with_suffix(".csv").write_text("step\n0\n")
    meta = {
        "run_id": name, "scenario": scenario, "seed": seed,
        "summary": {"success": int(success), "steps": steps, "deliver_step": deliver_step,
                    "return": ret, "n_rewinds": 0},
    }
    (tmp_path / f"{name}.meta.json").write_text(json.dumps(meta))


def test_load_and_group(tmp_path):
    _write_run(tmp_path, "heavy_s0", "heavy", 0, True, 120, 120)
    _write_run(tmp_path, "heavy_s1", "heavy", 1, False, 600, -1)
    _write_run(tmp_path, "fragile_s0", "fragile", 0, True, 100, 100)
    runs = load_runs(tmp_path)
    assert len(runs) == 3
    groups = group_by_scenario(runs)
    assert set(groups) == {"heavy", "fragile"}
    assert len(groups["heavy"]) == 2


def test_success_rate_and_episode_length(tmp_path):
    _write_run(tmp_path, "heavy_s0", "heavy", 0, True, 120, 118)
    _write_run(tmp_path, "heavy_s1", "heavy", 1, True, 130, 128)
    _write_run(tmp_path, "heavy_s2", "heavy", 2, False, 600, -1)
    m = scenario_metrics(load_runs(tmp_path))
    assert abs(m["success_rate"] - 2 / 3) < 1e-9          # 2 of 3 succeeded
    assert abs(m["mean_episode_length_success"] - 125.0) < 1e-9  # mean of 120,130
    assert m["best_steps_to_success"] == 118              # fewest steps-to-success among successes


def test_best_run_is_shortest_successful(tmp_path):
    _write_run(tmp_path, "a", "heavy", 0, True, 200, 195)
    _write_run(tmp_path, "b", "heavy", 1, True, 130, 128)   # shortest success -> winner
    _write_run(tmp_path, "c", "heavy", 2, True, 150, 149)
    best = best_per_scenario(load_runs(tmp_path))["heavy"]
    assert best.path.name == "b"


def test_no_success_returns_none(tmp_path):
    _write_run(tmp_path, "x", "fragile", 0, False, 600, -1)
    assert best_per_scenario(load_runs(tmp_path))["fragile"] is None


def test_rank_orders_by_success_then_length(tmp_path):
    # fragile: 1.0 success; heavy: 0.5 success -> fragile ranks first
    _write_run(tmp_path, "f0", "fragile", 0, True, 100, 100)
    _write_run(tmp_path, "f1", "fragile", 1, True, 110, 110)
    _write_run(tmp_path, "h0", "heavy", 0, True, 120, 120)
    _write_run(tmp_path, "h1", "heavy", 1, False, 600, -1)
    ranking = rank_scenarios(load_runs(tmp_path))
    assert [scn for scn, _ in ranking][0] == "fragile"
