# scripts/probe_spawn.py — diagnose the 100%-die-at-step-5 spawn failure.
#
# ENTRY SCRIPT: owns AppLauncher (env/ modules must not). Builds WarehouseGymEnv (1 env),
# resets, steps 6x with ZERO action (policy removed as a variable), and each step prints the
# RAW (un-grace-gated) failure state: chassis xyz/z, contact force, out-of-bounds, under-rack.
# If failure conditions are TRUE at spawn with zero action, the death is spawn-state, not policy.
#
# Usage:
#   python scripts/probe_spawn.py --headless --stage 3

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Probe the spawn-state failure")
parser.add_argument("--stage", type=int, default=3, help="Curriculum stage to mirror the run")
parser.add_argument("--steps", type=int, default=6, help="Zero-action steps to probe")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402

from env.warehouse_env import WarehouseEnvCfg, WarehouseGymEnv  # noqa: E402
from env.warehouse_reward import (  # noqa: E402
    RACK_HALF_X, RACK_HALF_Y, _contact_force, _rack_xy, _robot_xy,
)


def _chassis_z(env) -> float:
    """base_link world z (m) — catches floor penetration / wrong spawn height."""
    asset = env.scene["robot"]
    idx = asset.body_names.index("base_link")
    return float(asset.data.body_pos_w[0, idx, 2])


def _under_rack_dist(env) -> float:
    """Min over racks of the max-axis normalized footprint distance: <1.0 = chassis inside a rack."""
    xy = _robot_xy(env, SceneEntityCfg("robot"))[0]      # (2,)
    racks = _rack_xy(env)                                  # (R,2)
    dx = (xy[0] - racks[:, 0]).abs() / RACK_HALF_X
    dy = (xy[1] - racks[:, 1]).abs() / RACK_HALF_Y
    return float(torch.maximum(dx, dy).min())             # <1 = inside a rack footprint


def main() -> None:
    """Reset once, step zero action, print raw spawn-failure state each step."""
    cfg = WarehouseEnvCfg()
    cfg.scene.num_envs = 1
    gym = WarehouseGymEnv(cfg=cfg)
    gym._env.set_stage(args_cli.stage)
    print(f"[probe] stage={args_cli.stage}  zero-action x{args_cli.steps}", flush=True)

    gym.reset()
    env = gym._env
    zero = torch.zeros((1, 6), device=env.device)

    for t in range(args_cli.steps):
        xy = _robot_xy(env, SceneEntityCfg("robot"))[0]
        z = _chassis_z(env)
        cf = float(_contact_force(env, SceneEntityCfg("contact_sensor"))[0])
        oob = bool(xy[0].abs() > 9.5 or xy[1].abs() > 14.5)
        ur = _under_rack_dist(env)
        print(f"[probe] t={t} xy=({xy[0]:+.2f},{xy[1]:+.2f}) z={z:+.3f} "
              f"contactN={cf:.1f} OOB={oob} under_rack_min={ur:.2f}"
              f"{'  <-- INSIDE RACK' if ur < 1.0 else ''}"
              f"{'  <-- CRASH(>50N)' if cf > 50.0 else ''}", flush=True)
        gym.step(zero)

    gym.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
