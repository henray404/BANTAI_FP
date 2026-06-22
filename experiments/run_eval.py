# experiments/run_eval.py
# Person 5 — CLI entry for the headless CA-SLOPE eval harness.
#
# Runs entirely on a Mac (no Isaac Lab, no torch — numpy + stdlib only). Emits per-step trace +
# per-episode summary CSVs and prints a compact per-scenario table.
#
# Examples:
#   python experiments/run_eval.py                       # CA-SLOPE (category), all scenarios, 3 seeds
#   python experiments/run_eval.py --mode generic        # generic SLOPE control
#   python experiments/run_eval.py --mode none           # no shaping (vanilla base reward)
#   python experiments/run_eval.py --ablation            # run all 3 modes back-to-back (RQ2 sweep)
#   python experiments/run_eval.py --out training/results/eval --seeds 0 1 2 3 4

"""CLI: run the headless CA-SLOPE eval harness and write CSV traces + a summary table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.eval_harness import EpisodeResult, EvalHarness  # noqa: E402


def _print_table(mode: str, results: list[EpisodeResult]) -> None:
    """Print a per-scenario aggregate (success rate, mean steps, mean returns) across seeds."""
    print(f"\n=== mode={mode} ===")
    print(f"{'scenario':<16}{'n':>3}{'success':>9}{'mean_steps':>12}"
          f"{'mean_R_base':>13}{'mean_R_total':>14}{'mean_deliver':>14}")
    by_scn: dict[str, list[EpisodeResult]] = {}
    for r in results:
        by_scn.setdefault(r.scenario, []).append(r)
    for scn, rs in by_scn.items():
        n = len(rs)
        succ = sum(r.success for r in rs) / n
        steps = sum(r.steps for r in rs) / n
        rbase = sum(r.return_base for r in rs) / n
        rtot = sum(r.return_total for r in rs) / n
        delivered = [r.deliver_step for r in rs if r.deliver_step > 0]
        mdel = sum(delivered) / len(delivered) if delivered else float("nan")
        print(f"{scn:<16}{n:>3}{succ:>9.2f}{steps:>12.1f}{rbase:>13.2f}{rtot:>14.2f}{mdel:>14.1f}")


def main() -> None:
    """Parse args, run the harness for the requested mode(s), print tables."""
    p = argparse.ArgumentParser(description="Headless CA-SLOPE eval harness")
    p.add_argument("--mode", choices=["category", "generic", "none"], default="category",
                   help="CA-SLOPE category-aware, generic SLOPE, or no shaping")
    p.add_argument("--ablation", action="store_true",
                   help="run all three modes back-to-back (RQ2 sweep); overrides --mode")
    p.add_argument("--out", default="training/results/eval", help="output dir for CSVs")
    p.add_argument("--record_dir", default="", help="also record each episode as a replayable run "
                   "here (then rank with scripts/rank_runs.py)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="seeds per scenario")
    p.add_argument("--max_steps", type=int, default=600, help="episode step cap")
    p.add_argument("--run_id", default="toy", help="tag written into every CSV row")
    args = p.parse_args()

    modes = ["category", "generic", "none"] if args.ablation else [args.mode]
    for mode in modes:
        harness = EvalHarness(
            mode=mode, seeds=tuple(args.seeds), max_steps=args.max_steps, run_id=args.run_id,
        )
        results = harness.run(args.out, record_dir=args.record_dir or None)
        _print_table(mode, results)
    print(f"\n[OK] CSVs written under {Path(args.out).resolve()}")
    print("     steps_<mode>.csv = per-step trace · summary_<mode>.csv = per-episode metrics")


if __name__ == "__main__":
    main()
