# scripts/view_s4.py — open the s4 transfer scene in the Isaac GUI to LOOK at it (no training).
#
# Just run:  python scripts/view_s4.py
#
# It points the scene loader at configs/env_config_s4.yaml FOR YOU (sets $WAREHOUSE_ENV_CONFIG
# before importing the env), builds the full WarehouseGymEnv, and idles the robot so you can fly the
# camera around. You should see: a PHYSICALLY SMALLER rack, a box on each bottom shelf, AND a box on
# each 2nd (mid) shelf — fragile / regular / heavy. Nothing in the training env changes.
#
# Add --headless to run without a window (just a sanity print of the scene); omit it to SEE the GUI.
# UNVERIFIED on this hardware (Blackwell camera blocker) — best on the Linux/A100 box.

"""Spawn the s4 scene (smaller rack + 2nd-shelf boxes) and idle so you can look at it in the GUI."""

from __future__ import annotations

import os
from pathlib import Path

# Point the scene at the s4 config BEFORE importing the env (scene reads it at import time).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["WAREHOUSE_ENV_CONFIG"] = str(PROJECT_ROOT / "configs" / "env_config_s4.yaml")

import argparse  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="View the s4 transfer scene in the Isaac GUI")
parser.add_argument("--steps", type=int, default=100000,
                    help="How many physics steps to idle (keeps the window open). Ctrl-C to quit.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Project imports (after AppLauncher) ───────────────────────────────
import sys  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    """Build the s4 scene, reset, and idle with a zero action so the GUI stays interactive."""
    from env import warehouse_scene as ws
    print(f"[view_s4] scene override = {os.environ['WAREHOUSE_ENV_CONFIG']}")
    print(f"[view_s4] rack scale = {ws.RACK_SCALE}  (training = 0.01)")
    print(f"[view_s4] shelf levels = {tuple(round(z, 3) for z in ws.RACK_SHELF_LEVELS)}")
    print(f"[view_s4] boxes ({len(ws.TARGET_BOX_SPECS)}):")
    for name, size, _mass, pos in ws.TARGET_BOX_SPECS:
        print(f"    {name:18s} size={size:.2f}  z={pos[2]:.3f}")

    from env.warehouse_env import WarehouseEnvCfg, WarehouseGymEnv

    cfg = WarehouseEnvCfg()
    cfg.scene.num_envs = 1
    env = WarehouseGymEnv(cfg=cfg)
    try:
        env.reset()
        # Zero action each step = robot idles; sim keeps stepping so the GUI is live + boxes settle.
        act = np.zeros(env.action_space.shape, dtype=np.float32)
        for _ in range(args_cli.steps):
            if simulation_app.is_running() is False:
                break
            env.step(act)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
