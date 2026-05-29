# test_obs.py — Observation space verification.
#
# Usage:
#   python tests/test_obs.py --num_envs 1
#
# Prints obs key, shape, dtype, min/max. Intended for human inspection.

"""Dump WarehouseGymEnv observation tensor stats."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Warehouse env observation dump")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # warehouse env always uses onboard camera

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from env.warehouse_env import WarehouseEnvCfg, WarehouseGymEnv  # noqa: E402


def main() -> None:
    """Reset env, print obs key/shape/dtype/min/max."""
    cfg = WarehouseEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    env = WarehouseGymEnv(cfg=cfg)
    try:
        obs, _ = env.reset()
        print("\n=== Observation Space Dump ===")
        print(f"keys = {list(obs.keys())}")
        for key, val in obs.items():
            print(
                f"  {key}: shape={tuple(val.shape)} dtype={val.dtype} "
                f"min={float(val.min()):+.4f} max={float(val.max()):+.4f}"
            )
        print("\naction_space:", env.action_space)
        print("observation_space:", env.observation_space)
        print("\n=== DONE ===")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
