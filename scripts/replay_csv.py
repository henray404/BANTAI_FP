# scripts/replay_csv.py — faithfully replay a recorded run for the DEMO.
#
# Sets the recorded robot joints + target box pose into the sim each step (kinematic playback, no
# re-simulation), so the demo shows the EXACT recorded best run regardless of physics nondeterminism.
#
# Must run in the Isaac env (Windows box):
#   conda activate isaaclab
#   python scripts/replay_csv.py --run runs/best_run            # windowed, chase camera
#   python scripts/replay_csv.py --run runs/best_run --no_sleep # as fast as possible
#
# AppLauncher MUST be created before any isaaclab imports.

"""Replay a recorded scenario CSV faithfully in the real env (for demos)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Faithfully replay a recorded run CSV")
parser.add_argument("--run", required=True, help="run path (.csv / .meta.json / stem)")
parser.add_argument("--no_sleep", action="store_true", help="don't pace to control_dt (fast)")
parser.add_argument("--chase", type=int, default=1, help="3rd-person chase camera (1=on)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from env.warehouse_env import WarehouseEnvCfg, WarehouseGymEnv  # noqa: E402
from recording.recorder import TrajectoryReader  # noqa: E402
from recording.replay import apply_row_to_env, reset_to_recorded_scenario  # noqa: E402
from recording.state_extractor import _inner  # noqa: E402


def _chase_cam(env, row) -> None:
    """Point the viewport behind+above the recorded base pose (3rd-person follow)."""
    import math
    ie = _inner(env)
    bx, by, bz = row["base_x"], row["base_y"], row["base_z"]
    yaw = math.radians(row["base_yaw_deg"])
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    ie.sim.set_camera_view(eye=(bx - 6.0 * cos_y, by - 6.0 * sin_y, bz + 4.0),
                           target=(bx + cos_y, by + sin_y, bz + 0.5))


def main() -> None:
    """Build the env, seed to the recorded scenario, then set recorded state each step + render."""
    import time

    cfg = WarehouseEnvCfg()
    cfg.scene.num_envs = 1
    env = WarehouseGymEnv(cfg=cfg)

    reader = TrajectoryReader(args_cli.run)
    print(f"[replay] {reader.meta.get('category')} | {len(reader)} steps | "
          f"summary={reader.summary}")
    reset_to_recorded_scenario(env, reader)
    ie = _inner(env)
    dt = float(reader.meta.get("control_dt", 0.1))

    for row in reader.rows:
        if not simulation_app.is_running():
            break
        apply_row_to_env(env, row, reader.joint_names)
        if args_cli.chase:
            _chase_cam(env, row)
        ie.sim.render()
        if not args_cli.no_sleep:
            time.sleep(dt)
    print("[replay] done.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
