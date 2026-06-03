# drive_robot.py — Manual keyboard teleop for the Ridgeback-Franka base.
#
# Purpose: drive the robot by hand to confirm it loads + the base mapping works,
# WITHOUT the RL managers or the onboard camera (camera is stripped, so this avoids the
# Blackwell SDP crash — see bugs_errors/2026-05-22_sdp-camera-crash-blackwell.md).
#
# AppLauncher MUST be created here, before any isaaclab imports from env/.
# See bugs_errors/2026-05-15_double-applaunch-crash.md.
#
# Usage:
#   conda activate isaaclab
#   python scripts/drive_robot.py
#   python scripts/drive_robot.py --lin 1.5 --ang 1.5 --strafe 0.0
#
# Keys (window must be focused):
#   Arrow Up / Down    : forward / back   (base x)
#   Arrow Left / Right : strafe left/right (base y; default OFF to match the RL env)
#   Z / X              : yaw + / -
#   L                  : reset all commands to zero

"""Keyboard teleop: drive the Ridgeback-Franka base, camera-free (no SDP crash)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# ── CLI + AppLauncher ─────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Keyboard teleop for the warehouse robot")
parser.add_argument("--lin", type=float, default=1.5, help="forward/back speed sensitivity (m/s)")
parser.add_argument("--ang", type=float, default=1.5, help="yaw speed sensitivity (rad/s)")
parser.add_argument("--strafe", type=float, default=0.0, help="lateral speed (0 = match RL env, no strafe)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# No enable_cameras: the onboard TiledCamera is stripped below, so the SDP graph never inits.

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Imports (after AppLauncher) ───────────────────────────────────────
import torch

import isaaclab.sim as sim_utils
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from env.warehouse_scene import WarehouseSceneCfg  # noqa: E402

# Base joint order MUST match warehouse_env.ActionsCfg.base_vel: [vx, vy, wz].
BASE_JOINTS = [
    "dummy_base_prismatic_x_joint",
    "dummy_base_prismatic_y_joint",
    "dummy_base_revolute_z_joint",
]


def _build_scene() -> tuple[SimulationContext, InteractiveScene]:
    """Create sim + warehouse scene with camera and contact sensor stripped."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, render_interval=1)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(0.0, -12.0, 10.0), target=(0.0, 6.0, 0.5))
    scene_cfg = WarehouseSceneCfg(num_envs=1, env_spacing=22.0)
    scene_cfg.camera = None          # avoid Blackwell SDP crash; teleop needs no sensor cam
    scene_cfg.contact_sensor = None  # not needed for manual driving
    scene = InteractiveScene(scene_cfg)
    return sim, scene


def main() -> None:
    """Spawn robot, then loop: read keyboard SE(2) command -> base joint velocity targets."""
    sim, scene = _build_scene()
    sim.reset()
    robot = scene["robot"]

    # Resolve the 3 base joints in our fixed order (vx, vy, wz).
    base_ids, base_names = robot.find_joints(BASE_JOINTS, preserve_order=True)
    print(f"[drive] base joints (order = vx, vy, wz): {base_names} -> ids {base_ids}")
    print(f"[drive] robot bodies: {robot.body_names}")
    print(f"[drive] robot joints: {robot.joint_names}")

    keyboard = Se2Keyboard(
        Se2KeyboardCfg(
            v_x_sensitivity=args_cli.lin,
            v_y_sensitivity=args_cli.strafe,
            omega_z_sensitivity=args_cli.ang,
            sim_device=sim.device,
        )
    )
    print(keyboard)
    print("[drive] Focus the viewport window and use the arrow keys / Z / X to drive. Ctrl-C to quit.")

    while simulation_app.is_running():
        cmd = keyboard.advance().unsqueeze(0)            # (1, 3) = [vx, vy, wz]
        robot.set_joint_velocity_target(cmd, joint_ids=base_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt=sim.get_physics_dt())


if __name__ == "__main__":
    main()
    simulation_app.close()
