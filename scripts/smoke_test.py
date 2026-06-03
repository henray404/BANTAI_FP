# smoke_test.py — Automated (no-keyboard) base smoke test for the Ridgeback-Franka.
#
# Purpose: confirm the robot LOADS, log joint/body names, and PROGRAMMATICALLY answer
# Gotcha-B (are the base prismatic joints in WORLD frame or BODY frame?) by driving the
# base itself and measuring world-frame displacement. No human at the keyboard.
#
# Camera + contact sensor are stripped (avoids the Blackwell SDP crash, see
# bugs_errors/2026-05-22_sdp-camera-crash-blackwell.md). AppLauncher created here first
# (see bugs_errors/2026-05-15_double-applaunch-crash.md).
#
# Usage:
#   conda activate isaaclab
#   python scripts/smoke_test.py --headless

"""Automated base smoke test: load check + Gotcha-B (prismatic frame) verdict."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# ── CLI + AppLauncher ─────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Automated base smoke test for the warehouse robot")
parser.add_argument("--phase_steps", type=int, default=200, help="physics steps per drive phase")
parser.add_argument("--speed", type=float, default=1.0, help="forward command (m/s) on prismatic_x")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Imports (after AppLauncher) ───────────────────────────────────────
import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from env.warehouse_scene import WarehouseSceneCfg  # noqa: E402

# Base joint order = [vx, vy, wz]; must match warehouse_env.ActionsCfg.base_vel.
BASE_JOINTS = [
    "dummy_base_prismatic_x_joint",
    "dummy_base_prismatic_y_joint",
    "dummy_base_revolute_z_joint",
]


def _yaw_from_quat(quat_wxyz: torch.Tensor) -> float:
    """Return yaw (rad) from a single (w, x, y, z) quaternion tensor."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _build_scene() -> tuple[SimulationContext, InteractiveScene]:
    """Create sim + warehouse scene with camera and contact sensor stripped."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, render_interval=1)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(0.0, -12.0, 10.0), target=(0.0, 6.0, 0.5))
    scene_cfg = WarehouseSceneCfg(num_envs=1, env_spacing=22.0)
    scene_cfg.camera = None
    scene_cfg.contact_sensor = None
    scene = InteractiveScene(scene_cfg)
    return sim, scene


def _drive(sim, scene, robot, base_ids, cmd: list[float], steps: int) -> None:
    """Hold a base velocity command [vx, vy, wz] for N physics steps."""
    cmd_t = torch.tensor([cmd], device=sim.device)
    for _ in range(steps):
        robot.set_joint_velocity_target(cmd_t, joint_ids=base_ids)
        scene.write_data_to_sim()
        sim.step(render=False)  # headless, no camera -> skip render (huge speedup)
        scene.update(dt=sim.get_physics_dt())


def _base_xy(robot, body_idx: int) -> torch.Tensor:
    """Return env-0 world XY of the given body (the moving base_link, NOT the root).

    The Ridgeback-Franka articulation root is the FIXED `world` link, so root_pos_w never
    changes while the chassis translates via the prismatic joints. Read body_pos_w[base_link]
    or the frame verdict is always INCONCLUSIVE (disp ~0).
    """
    return robot.data.body_pos_w[0, body_idx, :2].clone()


def main() -> None:
    """Load robot, log names, drive forward / yaw / forward, print Gotcha-B verdict."""
    sim, scene = _build_scene()
    sim.reset()
    robot = scene["robot"]

    # ── Load + names ──────────────────────────────────────────────────
    base_ids, base_names = robot.find_joints(BASE_JOINTS, preserve_order=True)
    body_ids, _ = robot.find_bodies("base_link")
    bidx = int(body_ids[0])  # measure the MOVING chassis body, not the fixed `world` root
    print("\n========== SMOKE: LOAD CHECK ==========")
    print(f"[smoke] num joints: {robot.num_joints}  num bodies: {len(robot.body_names)}")
    print(f"[smoke] body names: {robot.body_names}")
    print(f"[smoke] joint names: {robot.joint_names}")
    print(f"[smoke] base joints (vx,vy,wz order): {base_names} -> ids {base_ids}")
    print(f"[smoke] base_link body idx: {bidx}")

    # Settle a few steps (arm holds tucked pose via position control).
    _drive(sim, scene, robot, base_ids, [0.0, 0.0, 0.0], 40)
    z0 = float(robot.data.body_pos_w[0, bidx, 2])
    yaw0 = _yaw_from_quat(robot.data.body_quat_w[0, bidx])
    print(f"[smoke] post-settle base_link height z = {z0:.3f} m  (sane if ~0.0-0.3)")

    # ── Phase 1: forward (prismatic_x = +speed), yaw fixed ───────────
    p0 = _base_xy(robot, bidx)
    _drive(sim, scene, robot, base_ids, [args_cli.speed, 0.0, 0.0], args_cli.phase_steps)
    p1 = _base_xy(robot, bidx)
    dir1 = (p1 - p0)
    d1 = float(torch.linalg.norm(dir1))

    # ── Phase 2: yaw +90 deg (revolute_z), no translation ────────────
    _drive(sim, scene, robot, base_ids, [0.0, 0.0, 0.8], args_cli.phase_steps)
    _drive(sim, scene, robot, base_ids, [0.0, 0.0, 0.0], 20)
    yaw2 = _yaw_from_quat(robot.data.body_quat_w[0, bidx])
    dyaw = math.degrees(yaw2 - yaw0)

    # ── Phase 3: forward again (same prismatic_x command) ─────────────
    p2 = _base_xy(robot, bidx)
    _drive(sim, scene, robot, base_ids, [args_cli.speed, 0.0, 0.0], args_cli.phase_steps)
    p3 = _base_xy(robot, bidx)
    dir2 = (p3 - p2)
    d2 = float(torch.linalg.norm(dir2))

    # ── Verdict ───────────────────────────────────────────────────────
    print("\n========== SMOKE: GOTCHA-B (prismatic frame) ==========")
    print(f"[smoke] phase1 forward disp: {dir1.tolist()}  |{d1:.3f} m|")
    print(f"[smoke] yaw applied phase2 : {dyaw:+.1f} deg")
    print(f"[smoke] phase3 forward disp: {dir2.tolist()}  |{d2:.3f} m|")
    if d1 < 0.02 or d2 < 0.02:
        print("[smoke] VERDICT: INCONCLUSIVE — base barely moved; check actuation/effort.")
    else:
        cos = float(torch.dot(dir1 / d1, dir2 / d2))
        if cos > 0.7:
            print(f"[smoke] VERDICT: WORLD-FRAME prismatic (cos={cos:.2f}). Forward ignores heading.")
            print("[smoke]   -> FIX base map: vx_world=v*cos(yaw); vy_world=v*sin(yaw); wz=omega.")
        elif abs(cos) < 0.4:
            print(f"[smoke] VERDICT: BODY-FRAME prismatic (cos={cos:.2f}). Forward follows heading.")
            print("[smoke]   -> current drive_robot.py / direct prismatic_x=v mapping is CORRECT.")
        else:
            print(f"[smoke] VERDICT: AMBIGUOUS (cos={cos:.2f}). Inspect manually.")
    print("========================================\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
