# scripts/replay_best.py — replay the BEST recorded episode in the Isaac Lab GUI.
#
# ENTRY SCRIPT: owns AppLauncher (see bugs_errors/2026-05-15_double-applaunch-crash.md).
#
# Loads a run's best/best_init.json (scene snapshot) + best/best_trajectory.csv (the model's
# per-step actions), restores the exact starting scene, then re-applies the recorded actions
# step by step so you SEE the best run in the GUI — no retraining, no agent reload.
#
# The init snapshot is restored BEFORE stepping because box poses + goal are randomized each
# reset; action-only replay would diverge. See env/scene_snapshot.py.
#
# Usage:
#   python scripts/replay_best.py --run training/results/ablation/c6_dreamer_full_seed0
#   python scripts/replay_best.py --run <dir> --headless          # no window (sanity check)

"""Replay the recorded best episode (actions + scene snapshot) in the Isaac Lab GUI."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay the best recorded episode in the GUI")
parser.add_argument("--run", type=str, required=True,
                    help="Run logdir containing best/best_trajectory.csv + best_init.json.")
parser.add_argument("--csv", type=str, default=None, help="Override trajectory CSV path.")
parser.add_argument("--init", type=str, default=None, help="Override init snapshot JSON path.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # env obs needs the camera even for replay

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Project imports (after AppLauncher) ───────────────────────────────
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from env.warehouse_env import WarehouseEnvCfg, WarehouseGymEnv  # noqa: E402
from env.scene_snapshot import restore_init_state  # noqa: E402
from experiments.metrics import episode_success  # noqa: E402


def _load(run: str, csv_override, init_override):
    """Return (init_snapshot dict, list of (6,) action arrays) for the run's best episode."""
    base = Path(run)
    csv_path = Path(csv_override) if csv_override else base / "best" / "best_trajectory.csv"
    init_path = Path(init_override) if init_override else base / "best" / "best_init.json"
    if not csv_path.exists() or not init_path.exists():
        raise FileNotFoundError(
            f"Missing replay files. Expected:\n  {csv_path}\n  {init_path}\n"
            "Run training first so the best episode is recorded."
        )
    meta = json.loads(init_path.read_text(encoding="utf-8"))
    actions = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            actions.append(np.array([float(row[f"a{i}"]) for i in range(6)], np.float32))
    return meta, actions


def main() -> None:
    """Build the env, restore the snapshot, and replay the recorded actions in the GUI."""
    meta, actions = _load(args_cli.run, args_cli.csv, args_cli.init)
    print(f"[replay] {len(actions)} steps | recorded success={meta.get('success_rate')} "
          f"return={meta.get('ep_return'):.2f}")

    cfg = WarehouseEnvCfg()
    cfg.scene.num_envs = 1
    env = WarehouseGymEnv(cfg=cfg)
    try:
        env.reset()
        restore_init_state(env._env, meta["init"])  # exact starting scene
        for i, action in enumerate(actions):
            env.step(action)
            if i % 50 == 0:
                print(f"[replay] step {i}/{len(actions)}")
        delivered = episode_success(env)
        print(f"[replay] done. delivered={delivered}")
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
