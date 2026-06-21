# experiments/run_all.py
# P5 — orchestrate the full 18-run ablation (6 configs x 3 seeds), sequentially.
#
# One GPU, one parallel env -> runs MUST be sequential (spec "Anggaran pelatihan").
# Each run is a fresh subprocess (an Isaac sim per process; also isolates the Blackwell
# close()-hang). Resumable: a run that finished writes a DONE marker and is skipped on
# re-invocation.
#
# Caveat (Blackwell / RTX 5050 only): a finished Isaac run can leave a zombie python.exe
# (close() hang). This orchestrator does NOT auto-kill processes (it would risk killing
# itself); if a run hangs past --timeout it is terminated and marked failed. Kill leftover
# python.exe manually between runs on Blackwell. See CLAUDE.md.
#
# Usage:
#   python -m experiments.run_all --dry-run                 # print the 18 commands
#   python -m experiments.run_all                           # run all (headless)
#   python -m experiments.run_all --only 3 4 5 6            # only the DreamerV3 configs
#   python -m experiments.run_all --seeds 0                 # one seed across all configs
#   python -m experiments.run_all --steps 5000             # short smoke budget

"""Sequential orchestrator for the 18-run 2x2 ablation + baselines."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from experiments.configs import CONFIGS, ExperimentConfig
from experiments.settings import load_settings


def build_command(cfg: ExperimentConfig, seed: int, logdir: Path, steps: int,
                  python: str, headless: bool, config_path: str | None) -> list[str]:
    """Build the subprocess argv for one (config, seed) run."""
    common = [python]
    head = ["--headless"] if headless else []
    extra = ["--config", config_path] if config_path else []
    if cfg.algo in ("sac", "ppo"):
        cmd = common + ["scripts/train_sac.py", "--algo", cfg.algo,
                        "--seed", str(seed), "--timesteps", str(steps),
                        "--logdir", str(logdir)] + head + extra
        if cfg.ca_slope:
            cmd.append("--ca_slope")
        return cmd
    # dreamer
    cmd = common + ["scripts/train_dreamer.py", "--seed", str(seed),
                    "--steps", str(steps), "--logdir", str(logdir)] + head + extra
    if cfg.ca_slope:
        cmd.append("--ca_slope")
    if cfg.visual_her:
        cmd.append("--visual_her")
    return cmd


def main() -> None:
    """Parse CLI, iterate the run schedule, launch each as a resumable subprocess."""
    ap = argparse.ArgumentParser(description="Run the 18-run ablation sequentially")
    ap.add_argument("--results", default="training/results/ablation")
    ap.add_argument("--only", type=int, nargs="*", default=None,
                    help="Restrict to these config indices (1..6).")
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="Override seeds (default: from --config / ablation.yaml).")
    ap.add_argument("--steps", type=int, default=None,
                    help="Override env steps/run (default: from --config / ablation.yaml).")
    ap.add_argument("--config", type=str, default=None,
                    help="Path to experiments/ablation.yaml; forwarded to every run.")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--timeout", type=int, default=None, help="Per-run timeout (seconds).")
    ap.add_argument("--no-headless", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    settings = load_settings(args.config)
    seeds = args.seeds if args.seeds is not None else settings.budget["seeds"]
    steps = args.steps if args.steps is not None else settings.budget["total_steps"]
    results = Path(args.results)
    configs = [c for c in CONFIGS if args.only is None or c.idx in args.only]
    schedule = [(c, s) for c in configs for s in seeds]

    print(f"[run_all] {len(schedule)} runs, {steps} steps each "
          f"-> {results}\n")
    failures = []
    for i, (cfg, seed) in enumerate(schedule, 1):
        logdir = results / f"{cfg.logname}_seed{seed}"
        done = logdir / "DONE"
        cmd = build_command(cfg, seed, logdir, steps, args.python,
                            not args.no_headless, args.config)
        tag = f"[{i}/{len(schedule)}] #{cfg.idx} {cfg.logname} seed{seed}"

        if done.exists():
            print(f"{tag}: SKIP (DONE marker present)")
            continue
        print(f"{tag}: {' '.join(cmd)}")
        if args.dry_run:
            continue

        logdir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, timeout=args.timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            print(f"{tag}: TIMEOUT after {args.timeout}s")
            rc = -1
        dt = time.time() - t0

        if rc == 0:
            done.write_text(f"ok steps={steps} seconds={dt:.0f}\n", encoding="utf-8")
            print(f"{tag}: OK ({dt:.0f}s)\n")
        else:
            failures.append((tag, rc))
            print(f"{tag}: FAILED rc={rc} ({dt:.0f}s)\n")

    print("\n[run_all] complete.")
    if failures:
        print(f"[run_all] {len(failures)} failed:")
        for tag, rc in failures:
            print(f"  {tag} (rc={rc})")
    print("[run_all] aggregate with:  python -m experiments.analyze "
          f"--results {results}")


if __name__ == "__main__":
    main()
