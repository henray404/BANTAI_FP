# demo_pickup.py — scripted full pickup→carry→sort demo (no training, no policy).
#
# Shows the magnetic-pickup behaviour end to end so you can SEE it work without training a policy:
#   drive base toward the commanded box  →  stop in front (arm is FROZEN, never reaches/knocks)  →
#   box is grabbed on proximity (invisible grip sized to the box)  →  carry to the colour-coded
#   zone (goal_id: fragile→orange, regular→cyan, heavy→purple)  →  env fires the delivered success.
#
# This is a hand-written base controller, NOT the RL policy — the real approach skill is learned in
# scripts/train_p3.py. It only steers [base_lin, base_ang]; the EE action is ignored (arm frozen by
# WarehouseGymEnv.step) and the gripper is held CLOSED so the magnetic grasp latches on proximity.
#
# AppLauncher MUST be created here, before any isaaclab imports from env/.
# See bugs_errors/2026-05-15_double-applaunch-crash.md.
#
# Usage:
#   conda activate isaaclab
#   python scripts/demo_pickup.py                  # Stage-2 (spawn near box): short clean approach
#   python scripts/demo_pickup.py --stage 3        # full chain (spawn north, long nav — may hit racks)
#   python scripts/demo_pickup.py --episodes 5

"""Scripted demo of the magnetic pickup → carry → colour-sort cycle."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Scripted pickup demo")
parser.add_argument("--stage", type=int, default=2, help="curriculum stage to demo (2 = spawn near box)")
parser.add_argument("--episodes", type=int, default=3, help="number of pickup episodes to run")
parser.add_argument("--max_steps", type=int, default=600, help="step cap per episode")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# The env mounts a TiledCamera, which requires the rendering pipeline — force cameras on.
# Set on args_cli (add_app_launcher_args already registered --enable_cameras); passing it as a
# kwarg too makes AppLauncher raise "both provided common attributes: {'enable_cameras'}".
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from env.warehouse_env import WarehouseEnvCfg, WarehouseGymEnv  # noqa: E402

DELIVER_RADIUS = 1.4   # consider the box delivered once the base is within this of the zone centre
FACE_TOL = 0.6         # only drive forward when the heading error is below this (rad)


def _f(t) -> np.ndarray:
    """First-env row of an obs tensor as a numpy float array."""
    return t[0].detach().cpu().numpy().astype(float)


def _yaw_err(target_xy, base_xy, heading) -> tuple[float, float]:
    """Return (distance, signed heading error) from base toward target_xy."""
    dx, dy = target_xy[0] - base_xy[0], target_xy[1] - base_xy[1]
    dist = math.hypot(dx, dy)
    desired = math.atan2(dy, dx)
    cur = math.atan2(heading[1], heading[0])           # heading = [cos(yaw), sin(yaw)]
    err = math.atan2(math.sin(desired - cur), math.cos(desired - cur))
    return dist, err


def _action(target_xy, base_xy, heading) -> np.ndarray:
    """Base controller: turn toward target, drive forward when facing it. Gripper held CLOSED."""
    dist, err = _yaw_err(target_xy, base_xy, heading)
    ang = float(np.clip(2.0 * err, -1.0, 1.0))
    lin = float(np.clip(1.0 * dist, 0.0, 1.0)) if abs(err) < FACE_TOL else 0.0
    # [base_lin, base_ang, ee_dx, ee_dy, ee_dz, gripper] — ee ignored (arm frozen), gripper<0 = closed.
    return np.array([lin, ang, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)


def _scalar(x) -> bool:
    """Truthiness of a possibly-batched terminated/truncated flag."""
    return bool(x[0]) if hasattr(x, "__len__") else bool(x)


def main() -> None:
    cfg = WarehouseEnvCfg()
    env = WarehouseGymEnv(cfg=cfg)
    env._env.set_stage(args_cli.stage)
    print(f"[demo] stage={args_cli.stage}  episodes={args_cli.episodes}")

    for ep in range(args_cli.episodes):
        obs, _ = env.reset()
        grabbed_at = None
        result = "timeout"
        for step in range(args_cli.max_steps):
            base_xy = _f(obs["position"])[:2]
            heading = _f(obs["heading"])
            holding = bool(_f(obs["holding"])[0] > 0.5)

            if not holding:
                target = _f(obs["box_pos"])[:2]            # phase A: go to the box
            else:
                if grabbed_at is None:
                    grabbed_at = step
                    print(f"[demo] ep{ep} GRABBED at step {step}")
                target = _f(obs["goal"])[:2]               # phase B: carry to the colour zone

            act = _action(target, base_xy, heading)
            obs, reward, terminated, truncated, info = env.step(act)

            if _scalar(terminated):
                result = "DELIVERED (success)" if holding else "terminated (bounds/other)"
                break
            if holding:
                dist_zone, _ = _yaw_err(_f(obs["goal"])[:2], _f(obs["position"])[:2], _f(obs["heading"]))
                if dist_zone < DELIVER_RADIUS:
                    result = "DELIVERED (in zone)"
                    break

        print(f"[demo] ep{ep} result: {result} (grabbed_at={grabbed_at})")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
