# drive_robot.py — Manual keyboard teleop for the Ridgeback-Franka base + arm (IK).
#
# Purpose: drive the robot by hand to confirm it loads, the base mapping works, AND the
# Franka arm tracks task-space (EE) commands through a DifferentialIKController — the same
# controller the RL env wires as its arm_ik action term. WITHOUT the RL managers or the
# onboard camera (camera is stripped, so this avoids the Blackwell SDP crash —
# see bugs_errors/2026-05-22_sdp-camera-crash-blackwell.md).
#
# AppLauncher MUST be created here, before any isaaclab imports from env/.
# See bugs_errors/2026-05-15_double-applaunch-crash.md.
#
# Usage:
#   conda activate isaaclab
#   python scripts/drive_robot.py
#   python scripts/drive_robot.py --lin 1.5 --ang 1.5 --strafe 0.0 --ee_sens 0.003
#
# Keys (window must be focused):
#   BASE:
#     Arrow Up / Down    : forward / back   (base x)
#     Arrow Left / Right : strafe left/right (base y; default OFF to match the RL env)
#     Z / X              : yaw + / -
#   ARM (end-effector, base frame, via IK):
#     W / S              : EE +x / -x   (forward / back)
#     A / D              : EE +y / -y   (left / right)
#     Q / E              : EE +z / -z   (up / down)
#     K                  : toggle gripper (open / close)
#   L                    : reset all commands to zero
#   (arm rotation keys T/G/C/V ignored — EE orientation held fixed top-down)

"""Keyboard teleop: drive the Ridgeback-Franka base + arm (IK), camera-free (no SDP crash)."""

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
parser.add_argument("--ee_sens", type=float, default=0.003,
                    help="EE position delta per physics step while an arm key is held (m). "
                         "Small because the loop runs at 200Hz; 0.003 ~= 0.6 m/s reach.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# No enable_cameras: the onboard TiledCamera is stripped below, so the SDP graph never inits.

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Imports (after AppLauncher) ───────────────────────────────────────
import torch

import isaaclab.sim as sim_utils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg, Se3Keyboard, Se3KeyboardCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import subtract_frame_transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from env.warehouse_scene import WarehouseSceneCfg  # noqa: E402

# Base joint order MUST match warehouse_env.ActionsCfg.base_vel: [vx, vy, wz].
BASE_JOINTS = [
    "dummy_base_prismatic_x_joint",
    "dummy_base_prismatic_y_joint",
    "dummy_base_revolute_z_joint",
]
# Arm / gripper — mirror warehouse_env.ActionsCfg.arm_ik + gripper terms.
ARM_JOINT_RE = "panda_joint.*"        # 7 revolute arm joints
FINGER_JOINT_RE = "panda_finger_joint.*"  # 2 prismatic fingers
EE_BODY = "panda_hand"                # Franka end-effector link
GRIP_OPEN, GRIP_CLOSE = 0.035, 0.0    # finger targets (match env open/close exprs)


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


def _ee_pose_base(robot, ee_body_idx) -> tuple[torch.Tensor, torch.Tensor]:
    """Current EE pose (pos (1,3), quat (1,4)) in the articulation root/base frame."""
    ee_w = robot.data.body_pose_w[:, ee_body_idx]        # (1, 7) pos+quat, world frame
    root_w = robot.data.root_pose_w                      # (1, 7) welded `world` link
    return subtract_frame_transforms(
        root_w[:, 0:3], root_w[:, 3:7], ee_w[:, 0:3], ee_w[:, 3:7]
    )


def _arm_ik_targets(robot, arm_ik, arm_ids, jac_joint_ids, ee_body_idx, ee_jacobi_idx,
                    ee_target, ee_quat_des) -> torch.Tensor:
    """Solve full-POSE IK to an ABSOLUTE EE target (base frame) -> arm joint targets (1, 7).

    Full 6-DOF pose (position + fixed orientation), absolute, for two reasons:
      * Absolute (not relative) so the arm HOLDS a fixed point — relative mode resets the
        target to the current (drooping) EE each step, ratcheting it down under gravity.
      * Pose (not position-only) so the 4-DOF redundancy of the 7-DOF arm is pinned by the
        orientation constraint; position-only IK lets the elbow wander / DLS jump configs.

    `jac_joint_ids` indexes the Jacobian DOF columns (may be offset by +6 from `arm_ids` when
    PhysX reports a FLOATING-base Jacobian); `arm_ids` indexes joint_pos / joint targets.
    """
    jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, jac_joint_ids]
    ee_pos_b, ee_quat_b = _ee_pose_base(robot, ee_body_idx)
    joint_pos = robot.data.joint_pos[:, arm_ids]
    command = torch.cat([ee_target, ee_quat_des], dim=-1)  # (1, 7) [x,y,z, qw,qx,qy,qz]
    arm_ik.set_command(command)                            # absolute pose
    return arm_ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)


def main() -> None:
    """Spawn robot, then loop: keyboard -> base joint velocities + arm IK joint targets + gripper."""
    sim, scene = _build_scene()
    sim.reset()
    robot = scene["robot"]

    # Resolve the 3 base joints in our fixed order (vx, vy, wz).
    base_ids, base_names = robot.find_joints(BASE_JOINTS, preserve_order=True)
    base_link_idx = robot.body_names.index("base_link")  # moving chassis (yaw source; #1268)
    print(f"[drive] base joints (order = vx, vy, wz): {base_names} -> ids {base_ids}")
    print(f"[drive] robot bodies: {robot.body_names}")
    print(f"[drive] robot joints: {robot.joint_names}")

    # Resolve arm + gripper, and build the same IK controller the RL env uses (position, relative).
    arm_ids, arm_names = robot.find_joints(ARM_JOINT_RE, preserve_order=True)
    finger_ids, _ = robot.find_joints(FINGER_JOINT_RE, preserve_order=True)
    ee_body_idx = robot.body_names.index(EE_BODY)
    # Resolve Jacobian indexing ROBUSTLY. The Ridgeback base is welded at runtime via an added
    # FixedJoint, so PhysX may still report a FLOATING-base Jacobian (6 extra root DOF columns)
    # even though is_fixed_base reads True. With a floating Jacobian, naive joint-id slicing picks
    # the WRONG columns and the IK diverges (EE flails, never reaches target — the bug we hit).
    # Detect from the Jacobian DOF width: floating => width == num_joints + 6.
    jac = robot.root_physx_view.get_jacobians()
    floating_jac = jac.shape[-1] == robot.num_joints + 6
    jac_joint_ids = [i + 6 for i in arm_ids] if floating_jac else list(arm_ids)
    # Jacobian body rows: floating includes the root row (no -1); fixed excludes it (-1).
    ee_jacobi_idx = ee_body_idx if floating_jac else ee_body_idx - 1
    arm_ik = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        num_envs=1, device=sim.device,
    )
    arm_ik.reset()
    print(f"[drive] arm joints: {arm_names} -> ids {arm_ids}  EE={EE_BODY}[{ee_body_idx}]")
    print(f"[drive] jacobian shape={tuple(jac.shape)} num_joints={robot.num_joints} "
          f"floating_jac={floating_jac} ee_jacobi_idx={ee_jacobi_idx} jac_joint_ids={jac_joint_ids} "
          f"is_fixed_base={robot.is_fixed_base}")

    keyboard = Se2Keyboard(
        Se2KeyboardCfg(
            v_x_sensitivity=args_cli.lin,
            v_y_sensitivity=args_cli.strafe,
            omega_z_sensitivity=args_cli.ang,
            sim_device=sim.device,
        )
    )
    # Arm teleop: W/S/A/D/Q/E -> EE xyz delta, K -> gripper toggle. Rotation disabled (top-down).
    arm_kb = Se3Keyboard(
        Se3KeyboardCfg(
            pos_sensitivity=args_cli.ee_sens,
            rot_sensitivity=0.0,
            gripper_term=True,
            sim_device=sim.device,
        )
    )
    print(keyboard)
    print("[drive] Focus the viewport. BASE: arrows + Z/X. ARM: W/S A/D Q/E, K=gripper. Ctrl-C to quit.")

    # Settle so the arm reaches its tucked spawn pose, then capture the HOLD target: the absolute
    # EE pose the arm parks at when no arm key is pressed. Keyboard deltas accumulate onto this.
    for _ in range(30):
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt=sim.get_physics_dt())
    ee_target, ee_quat_des = _ee_pose_base(robot, ee_body_idx)
    ee_target = ee_target.clone()
    ee_quat_des = ee_quat_des.clone()
    # Clamp box: a +/-0.4 m cube around the spawn EE pose, well inside the Franka reach (~0.85 m).
    ee_min = ee_target - 0.4
    ee_max = ee_target + 0.4
    print(f"[drive] arm hold target (base frame) = {[round(v, 3) for v in ee_target[0].tolist()]}")

    step = 0
    while simulation_app.is_running():
        # ── Base: SE(2) command -> world-frame prismatic velocities (yaw-rotated, #2664) ──
        cmd = keyboard.advance().unsqueeze(0)            # (1, 3) = [vx, vy, wz] body-frame intent
        q = robot.data.body_quat_w[0, base_link_idx]     # (w, x, y, z)
        yaw = torch.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2))
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        vx_b, vy_b = cmd[0, 0].clone(), cmd[0, 1].clone()
        cmd[0, 0] = vx_b * cy - vy_b * sy                # world-x velocity (prismatic_x)
        cmd[0, 1] = vx_b * sy + vy_b * cy                # world-y velocity (prismatic_y)
        robot.set_joint_velocity_target(cmd, joint_ids=base_ids)

        # ── Arm: accumulate SE(3) EE delta into an ABSOLUTE hold target -> IK -> joint targets ──
        arm_cmd = arm_kb.advance()                       # (7,) [dx,dy,dz, rx,ry,rz, grip]
        ee_delta = arm_cmd[:3].unsqueeze(0)              # (1, 3) base-frame position delta
        ee_target += ee_delta                            # held keys move the target; release holds it
        # Clamp the target to the Franka reach so it can't run past the workspace; once the target
        # leaves reach the IK saturates and the EE looks "stuck" / stops tracking the key.
        ee_target = torch.clamp(ee_target, min=ee_min, max=ee_max)
        arm_targets = _arm_ik_targets(
            robot, arm_ik, arm_ids, jac_joint_ids, ee_body_idx, ee_jacobi_idx, ee_target, ee_quat_des
        )
        robot.set_joint_position_target(arm_targets, joint_ids=arm_ids)
        finger = GRIP_OPEN if float(arm_cmd[-1]) > 0.0 else GRIP_CLOSE
        robot.set_joint_position_target(
            torch.full((1, len(finger_ids)), finger, device=sim.device), joint_ids=finger_ids
        )

        scene.write_data_to_sim()
        sim.step()
        scene.update(dt=sim.get_physics_dt())

        # Diagnostic (throttled ~5 Hz, silent when idle): bisects "robot won't move".
        #   - nothing prints      -> viewport window NOT focused (carb keyboard gets no events)
        #   - cmd != 0, vel ~= 0  -> command reaches joints but drive/physics not tracking
        #   - cmd != 0, vel != 0  -> working (look at the viewport, robot IS moving)
        if step % 40 == 0:
            cmd_l = [round(v, 3) for v in cmd[0].tolist()]
            base_vel = [round(v, 3) for v in robot.data.joint_vel[0, base_ids].tolist()]
            ee_now, _ = _ee_pose_base(robot, ee_body_idx)       # actual EE (base frame)
            err = float(torch.norm(ee_target - ee_now))         # hold error: should stay ~0
            tgt = [round(v, 3) for v in ee_target[0].tolist()]
            now = [round(v, 3) for v in ee_now[0].tolist()]
            if any(abs(v) > 1e-6 for v in cmd_l + base_vel) or err > 1e-3:
                print(f"[drive] base(vx,vy,wz)={cmd_l} vel={base_vel}  "
                      f"ee_target={tgt} ee_now={now} hold_err={err:.4f}m")
        step += 1


if __name__ == "__main__":
    main()
    simulation_app.close()
