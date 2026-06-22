#!/usr/bin/env python3
# scripts/rank_runs.py — score recorded runs per scenario and print the best demo run for each.
#
# Pure stdlib (no Isaac / torch) — runs on a Mac. Reads the <name>.meta.json sidecars written by the
# recorder (scripts/record_scenario.py, scripts/drive_env.py --record, or the toy eval harness).
#
#   python scripts/rank_runs.py --dir runs/
#   python scripts/rank_runs.py --dir runs/ --copy_best docs/demo   # also copy each best run there

"""CLI: rank recorded runs per scenario by success / efficiency / length; show the best per scenario."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from recording.select_best import best_per_scenario, load_runs, rank_scenarios  # noqa: E402


def main() -> None:
    """Scan a run directory, print the per-scenario ranking + best run, optionally copy the winners."""
    p = argparse.ArgumentParser(description="Rank recorded runs per scenario; pick the best demo run")
    p.add_argument("--dir", default="runs", help="directory of recorded runs (scanned recursively)")
    p.add_argument("--copy_best", default="", help="if set, copy each scenario's best CSV+meta here")
    args = p.parse_args()

    runs = load_runs(args.dir)
    if not runs:
        print(f"[rank] no runs (*.meta.json) found under {args.dir}")
        return

    print(f"[rank] {len(runs)} runs under {args.dir}\n")
    print(f"{'scenario':<18}{'runs':>5}{'success_rate':>14}{'best_steps':>12}"
          f"{'mean_len':>10}  best_run")
    ranking = rank_scenarios(runs)
    for scn, m in ranking:
        best = m["best"]
        best_name = best.path.name if best else "— (no success)"
        mean_len = m["mean_episode_length_success"]
        mean_len_s = f"{mean_len:.1f}" if mean_len == mean_len else "—"   # NaN guard
        best_steps = m["best_steps_to_success"]
        best_steps_s = str(best_steps) if best_steps is not None else "—"
        print(f"{scn:<18}{m['n_runs']:>5}{m['success_rate']:>14.2f}{best_steps_s:>12}"
              f"{mean_len_s:>10}  {best_name}")

    print("\n[rank] scenario winners (best demo run each):")
    winners = best_per_scenario(runs)
    for scn, best in winners.items():
        if best is None:
            print(f"  {scn:<18} — no successful run; record more seeds")
        else:
            print(f"  {scn:<18} {best.path}.csv  (steps={best.steps_to_success}, "
                  f"return={best.return_:.2f}, rewinds={best.n_rewinds})")

    # Overall best scenario = top of the ranking (highest success_rate, then shortest).
    top_scn = ranking[0][0] if ranking else None
    if top_scn:
        print(f"\n[rank] strongest scenario: {top_scn}")

    if args.copy_best:
        dest = Path(args.copy_best)
        dest.mkdir(parents=True, exist_ok=True)
        for scn, best in winners.items():
            if best is None:
                continue
            for suffix in (".csv", ".meta.json"):
                src = Path(str(best.path) + suffix)
                if src.exists():
                    shutil.copy(src, dest / src.name)
        print(f"\n[rank] copied scenario-winning runs -> {dest}")


if __name__ == "__main__":
    main()
