# recording/select_best.py
# Person 5 — score recorded runs per scenario and pick the best one for the demo.
#
# Metrics (per the P5 / dosen brief), computed PER SCENARIO (scenario = box category here):
#   (1) success_rate     : fraction of recorded episodes that SUCCEEDED, where success = the correct
#                          category box reached the correct color zone and was released. The env/
#                          recorder already encodes this in summary.success (delivery only fires on the
#                          matching zone), so success_rate = mean(success) over a scenario's runs.
#   (2) sample_efficiency: env steps to reach success. Per single demo run this is "steps-to-success"
#                          (deliver_step); fewer = more efficient. (The training-curve version "steps to
#                          reach a success-rate threshold" lives in the eval harness / W&B, not here —
#                          that needs learning curves, not recorded demo runs.)
#   (3) episode_length   : steps to completion (summary.steps).
#
# Best run PER SCENARIO = a SUCCESSFUL run with the fewest steps-to-success (tie-break: higher return).
# Pure stdlib → runs and is tested on a Mac.

"""Rank recorded runs per scenario by success / sample-efficiency / length and pick the best."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class RunScore:
    """One recorded run reduced to its ranking metrics (read from <name>.meta.json)."""

    path: Path                  # the run stem (…/<name>); CSV is <stem>.csv
    scenario: str               # category / scenario tag
    seed: Optional[int]
    success: bool
    steps: int                  # episode length (steps to completion)
    steps_to_success: int       # deliver_step if delivered, else steps (sample-efficiency proxy)
    return_: float
    n_rewinds: int


def load_runs(root: str | Path) -> list[RunScore]:
    """Load every run under `root` (recursively) from its <name>.meta.json sidecar."""
    out: list[RunScore] = []
    for meta_path in sorted(Path(root).rglob("*.meta.json")):
        with open(meta_path) as f:
            meta = json.load(f)
        summary = meta.get("summary", {})
        success = bool(summary.get("success", 0))
        steps = int(summary.get("steps", 0))
        deliver = int(summary.get("deliver_step", -1))
        stem = meta_path.with_name(meta_path.name[: -len(".meta.json")])
        out.append(RunScore(
            path=stem,
            scenario=str(meta.get("scenario") or meta.get("category") or meta.get("run_id", "unknown")),
            seed=meta.get("seed"),
            success=success,
            steps=steps,
            steps_to_success=deliver if (success and deliver > 0) else steps,
            return_=float(summary.get("return", 0.0)),
            n_rewinds=int(summary.get("n_rewinds", 0)),
        ))
    return out


def group_by_scenario(runs: list[RunScore]) -> dict[str, list[RunScore]]:
    """Bucket runs by their scenario tag."""
    groups: dict[str, list[RunScore]] = {}
    for r in runs:
        groups.setdefault(r.scenario, []).append(r)
    return groups


def scenario_metrics(runs: list[RunScore]) -> dict:
    """Aggregate one scenario's runs into the three metrics + the best run."""
    n = len(runs)
    successes = [r for r in runs if r.success]
    success_rate = len(successes) / n if n else 0.0
    best = best_run(runs)
    mean_len = sum(r.steps for r in successes) / len(successes) if successes else float("nan")
    return {
        "n_runs": n,
        "success_rate": success_rate,           # metric (1)
        "best_steps_to_success": best.steps_to_success if best else None,  # metric (2)
        "mean_episode_length_success": mean_len,                            # metric (3)
        "best": best,
    }


def best_run(runs: list[RunScore]) -> Optional[RunScore]:
    """Best run in a scenario: successful, fewest steps-to-success, tie-break higher return."""
    successful = [r for r in runs if r.success]
    if not successful:
        return None
    return min(successful, key=lambda r: (r.steps_to_success, -r.return_))


def rank_scenarios(runs: list[RunScore]) -> list[tuple[str, dict]]:
    """Per-scenario metrics, ordered best scenario first (success_rate desc, then shorter episodes)."""
    items = [(scn, scenario_metrics(rs)) for scn, rs in group_by_scenario(runs).items()]
    items.sort(key=lambda kv: (-kv[1]["success_rate"],
                               kv[1]["mean_episode_length_success"]
                               if kv[1]["mean_episode_length_success"] == kv[1]["mean_episode_length_success"]
                               else float("inf")))
    return items


def best_per_scenario(runs: list[RunScore]) -> dict[str, Optional[RunScore]]:
    """The single best demo run for each scenario (None if that scenario never succeeded)."""
    return {scn: best_run(rs) for scn, rs in group_by_scenario(runs).items()}
