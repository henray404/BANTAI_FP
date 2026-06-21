# tune_arm.py — P4 arm-reach verification + EE/grasp constant tuning harness (items 1 & 2).
#
# Purpose (spec §9): the RL env's two grasp constants are still at first-guess values —
#   EE_STEP_M    (env/action_pickup.py) = 0.05   # EE travel per control step at action=1.0
#   GRIP_RADIUS_M(env/grasp.py)         = 0.10   # EE-to-box-surface contact radius
# and "arm reach end-to-end" was never verified after the 2026-06-16 arm fixes. This script
# drives the Franka arm (same DifferentialIK the RL env wires) to a box of EACH size category,
# measures how close the EE actually gets to the box SURFACE, and sweeps candidate GRIP_RADIUS_M
# values so you can pick one that grasps every category WITHOUT false-positives. It also prints
# the resolved panda_hand / box prim paths the physics-grasp weld (env/attach.py) depends on.
#
# This is a DIAGNOSTIC you run in sim; it changes no source. Paste the summary table back and
# P4 updates EE_STEP_M / GRIP_RADIUS_M accordingly.
#
# AppLauncher MUST be created here, before any isaaclab imports from env/.
# See bugs_errors/2026-05-15_double-applaunch-crash.md.
#
# Usage:
#   conda activate isaaclab
#   python scripts/tune_arm.py                 # headless-ish (viewport opens), runs all 3 categories
#   python scripts/tune_arm.py --settle 40 --reach_steps 200 --standoff 0.8

"""Automated arm-reach + grasp-radius tuning harness for the warehouse pickup task."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# ── CLI + AppLauncher ─────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Arm-reach + grasp-radius tuning harness")
parser.add_argument("--settle", type=int, default=30, help="physics steps to settle after a base teleport")
parser.add_argument("--reach_steps", type=int, default=200, help="IK steps to drive the EE toward the box")
parser.add_argument("--standoff", type=float, default=0.8, help="base distance from the box (m)")
parser.add_argument("--standoffs", type=str, default="0.7,0.8,0.9,1.0",
                    help="comma list of base distances to sweep per category (finds the reachable one). "
                         "Keep >= ~0.7 — closer and the base footprint collides with the box.")
parser.add_argument("--ee_step", type=float, default=0.01,
                    help="per-step EE target advance toward the box (m); proxy for EE_STEP_M tuning")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# No enable_cameras: the onboard TiledCamera is stripped below (avoids Blackwell SDP crash).

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Imports (after AppLauncher) ───────────────────────────────────────
import torch

import isaaclab.sim as sim_utils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import subtract_frame_transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from env.warehouse_scene import TARGET_BOX_SPECS, WarehouseSceneCfg  # noqa: E402
from env.curriculum import spawn_pose_near_box  # noqa: E402
from env.grasp import GRIP_RADIUS_M  # noqa: E402
from env.action_pickup import EE_STEP_M  # noqa: E402
from env.attach import find_descendant_path  # noqa: E402

ARM_JOINT_RE = "panda_joint.*"
FINGER_JOINT_RE = "panda_finger_joint.*"
EE_BODY = "panda_hand"
BASE_JOINTS = [
    "dummy_base_prismatic_x_joint",
    "dummy_base_prismatic_y_joint",
    "dummy_base_revolute_z_joint",
]
GRIP_CLOSE = 0.0
RADIUS_SWEEP = (0.05, 0.08, 0.10, 0.12, 0.15)


def _build_scene() -> tuple[SimulationContext, InteractiveScene]:
    """Sim + warehouse scene with camera/contact sensor stripped (diagnostic, no perception)."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, render_interval=4)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(0.0, -6.0, 6.0), target=(0.0, 0.0, 0.5))
    scene_cfg = WarehouseSceneCfg(num_envs=1, env_spacing=32.0)
    scene_cfg.camera = None          # avoid Blackwell SDP crash; tuning needs no sensor cam
    scene_cfg.contact_sensor = None
    scene = InteractiveScene(scene_cfg)
    return sim, scene


def _ee_pose_base(robot, ee_idx) -> tuple[torch.Tensor, torch.Tensor]:
    """EE pose (pos (1,3), quat (1,4)) in the articulation root/base frame."""
    ee_w = robot.data.body_pose_w[:, ee_idx]
    root_w = robot.data.root_pose_w
    return subtract_frame_transforms(root_w[:, 0:3], root_w[:, 3:7], ee_w[:, 0:3], ee_w[:, 3:7])


def _world_to_base(robot, pos_w: torch.Tensor) -> torch.Tensor:
    """Express a world-frame point (1,3) in the robot root/base frame."""
    root_w = robot.data.root_pose_w
    quat_id = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=pos_w.device)
    p_b, _ = subtract_frame_transforms(root_w[:, 0:3], root_w[:, 3:7], pos_w, quat_id)
    return p_b


def _step(sim, scene, n: int) -> None:
    for _ in range(n):
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt=sim.get_physics_dt())


def _teleport_base(robot, scene, world_x: float, world_y: float, yaw: float, base_ids, device) -> None:
    """Write robot root pose to (world_x, world_y, yaw) and zero the dummy base joints.

    Mirrors WarehouseRLEnv._spawn_base_near_box so what we tune here matches Stage-2 reset.
    """
    import math

    root = robot.data.root_state_w.clone()
    root[0, 0] = world_x
    root[0, 1] = world_y
    half = yaw * 0.5
    root[0, 3] = math.cos(half)
    root[0, 4] = 0.0
    root[0, 5] = 0.0
    root[0, 6] = math.sin(half)
    root[0, 7:13] = 0.0
    robot.write_root_state_to_sim(root, env_ids=torch.tensor([0], device=device))
    zeros = torch.zeros(1, len(base_ids), device=device)
    robot.write_joint_state_to_sim(zeros, zeros, joint_ids=base_ids, env_ids=torch.tensor([0], device=device))


def main() -> None:  # noqa: C901 — diagnostic driver, linear top-to-bottom for readability
    sim, scene = _build_scene()
    sim.reset()
    robot = scene["robot"]
    device = sim.device

    base_ids, _ = robot.find_joints(BASE_JOINTS, preserve_order=True)
    arm_ids, _ = robot.find_joints(ARM_JOINT_RE, preserve_order=True)
    finger_ids, _ = robot.find_joints(FINGER_JOINT_RE, preserve_order=True)
    ee_idx = robot.body_names.index(EE_BODY)

    # Robust Jacobian indexing (welded base may still report a FLOATING Jacobian) — see drive_robot.py.
    jac = robot.root_physx_view.get_jacobians()
    floating = jac.shape[-1] == robot.num_joints + 6
    jac_joint_ids = [i + 6 for i in arm_ids] if floating else list(arm_ids)
    ee_jacobi_idx = ee_idx if floating else ee_idx - 1

    arm_ik = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        num_envs=1, device=device,
    )
    arm_ik.reset()

    env_origin = scene.env_origins[0]
    # One target box per category (size encodes category): smallest, medium, largest.
    by_size: dict[float, tuple] = {}
    for name, size, mass, pos in TARGET_BOX_SPECS:
        by_size.setdefault(size, (name, size, mass, pos))
    categories = sorted(by_size.values(), key=lambda t: t[1])  # fragile, regular, heavy

    # Prim-path VERIFY for the physics-grasp weld (env/attach.py).
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    hand_path = find_descendant_path(stage, "/World/envs/env_0/Robot", EE_BODY)
    print("\n========== PHYSICS-GRASP PRIM PATHS (env/attach.py) ==========")
    print(f"  panda_hand prim : {hand_path}   <- must be non-None for physics grasp")
    print(f"  box prim pattern: /World/envs/env_0/<box_name>")

    print("\n========== ARM-REACH + GRASP-RADIUS SWEEP ==========")
    print(f"  current EE_STEP_M={EE_STEP_M}  GRIP_RADIUS_M={GRIP_RADIUS_M}  standoff={args_cli.standoff}m")
    header = f"{'category':9} {'size':>5} {'min_surf_dist':>13} {'reach?':>7} " + " ".join(
        f"r={r:>4}" for r in RADIUS_SWEEP
    )
    print(header)

    # Capture the clean spawn state ONCE so every trial starts identical (no carryover / no
    # box getting shoved by a previous close-standoff collision — that corrupted the first sweep).
    _step(sim, scene, 60)                                   # let boxes settle onto the floor first
    default_joint_pos = robot.data.default_joint_pos.clone()
    box_spawn = {name: scene[name].data.root_state_w.clone() for name, *_ in categories}

    def _measure_reach(box, name: str, box_half: float, pos, standoff: float) -> float:
        """Reset box+arm, teleport base `standoff` m north of the box, IK-reach its NEAR FACE.

        Aims at the box face closest to the robot at CENTER height (not the top) — that is what the
        grasp model rewards (EE within GRIP_RADIUS_M of any surface) and is reachable on tall boxes
        that the arm cannot descend onto from above. Returns min EE→surface distance over the run.
        """
        # 1) restore arm to tucked spawn pose + box to its rest pose (kill carryover/collision).
        robot.write_joint_state_to_sim(default_joint_pos, torch.zeros_like(default_joint_pos))
        box.write_root_state_to_sim(box_spawn[name], env_ids=torch.tensor([0], device=device))
        _step(sim, scene, 10)
        # 2) place the base standoff m north of the box, facing it (Stage-2 geometry).
        bx, by = float(pos[0]), float(pos[1])
        base_x, base_y, yaw = spawn_pose_near_box((bx, by), standoff=standoff)
        _teleport_base(robot, scene, float(env_origin[0]) + base_x, float(env_origin[1]) + base_y,
                       yaw, base_ids, device)
        _step(sim, scene, args_cli.settle)
        # 3) IK target = near (+y) face of the box at center height, spawn EE orientation held.
        box_w = box.data.root_pos_w[0:1]
        target_w = box_w.clone()
        target_w[0, 1] += box_half                             # +y face (robot is north of box)
        _, ee_quat_des = _ee_pose_base(robot, ee_idx)
        ee_quat_des = ee_quat_des.clone()
        cur_target, _ = _ee_pose_base(robot, ee_idx)
        cur_target = cur_target.clone()
        min_surf = float("inf")
        for _ in range(args_cli.reach_steps):
            goal_b = _world_to_base(robot, target_w)
            delta = goal_b - cur_target
            dist = float(torch.norm(delta))
            step_vec = delta if dist <= args_cli.ee_step else delta * (args_cli.ee_step / dist)
            cur_target = cur_target + step_vec
            ee_pos_b, ee_quat_b = _ee_pose_base(robot, ee_idx)
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, jac_joint_ids]
            command = torch.cat([cur_target, ee_quat_des], dim=-1)
            arm_ik.set_command(command)
            arm_targets = arm_ik.compute(ee_pos_b, ee_quat_b, jacobian, robot.data.joint_pos[:, arm_ids])
            robot.set_joint_position_target(arm_targets, joint_ids=arm_ids)
            robot.set_joint_position_target(
                torch.full((1, len(finger_ids)), GRIP_CLOSE, device=device), joint_ids=finger_ids
            )
            _step(sim, scene, 1)
            surf = float(torch.norm(robot.data.body_pos_w[:, ee_idx] - box_w)) - box_half
            min_surf = min(min_surf, surf)
        return min_surf

    standoffs = [float(s) for s in args_cli.standoffs.split(",")]
    print(f"  standoff sweep: {standoffs}  (target = near face @ center height)")
    print(f"{'category':9} {'size':>5} " + " ".join(f"so={s:>4}" for s in standoffs) +
          f" {'best_so':>8} {'min_surf':>9} {'reach?':>7}")

    results = []
    for name, size, mass, pos in categories:
        box = scene[name]
        box_half = size / 2.0
        per_so = {s: _measure_reach(box, name, box_half, pos, s) for s in standoffs}
        best_so = min(per_so, key=per_so.get)
        best_surf = per_so[best_so]
        reach_ok = best_surf < max(RADIUS_SWEEP)
        cat = "fragile" if size < 0.25 else "regular" if size < 0.4 else "heavy"
        cols = " ".join(f"{per_so[s]:>7.3f}" for s in standoffs)
        print(f"{cat:9} {size:>5.2f} {cols} {best_so:>8.2f} {best_surf:>9.3f} "
              f"{('YES' if reach_ok else 'NO'):>7}")
        results.append((name, size, best_surf, best_so, reach_ok))

    print("\n========== RECOMMENDATION ==========")
    worst = max((r[2] for r in results), default=float("inf"))
    if all(r[4] for r in results):
        need_so = max(r[3] for r in results)  # largest standoff still reachable for ALL = safest single value
        print(f"  All categories reachable. Use Stage-2 standoff <= {min(r[3] for r in results):.2f} m "
              f"(every category reaches at its best_so column).")
        print(f"  Min GRIP_RADIUS_M that grasps every box: >= {worst:.3f} m "
              f"(round up, e.g. {((worst // 0.01) + 1) * 0.01:.2f}).")
        print(f"  Current GRIP_RADIUS_M={GRIP_RADIUS_M}: "
              f"{'OK' if GRIP_RADIUS_M >= worst else 'TOO SMALL — increase to the value above'}.")
        print(f"  Set spawn_pose_near_box default standoff (env/curriculum.py) accordingly "
              f"(currently 0.8; biggest box needs ~{need_so:.2f}).")
    else:
        bad = [f"{('fragile' if s < 0.25 else 'regular' if s < 0.4 else 'heavy')}({m:.3f}m)"
               for _, s, m, _so, ok in results if not ok]
        print(f"  Still NOT reachable even at closest standoff: {', '.join(bad)}.")
        print("  Try smaller --standoffs (e.g. 0.35,0.4,0.45) OR the reach limit is the box HEIGHT "
              "(top-down IK orientation can't descend onto a tall box) — needs a side-approach pose.")
    print(f"  EE_STEP_M: per-step advance capped at --ee_step={args_cli.ee_step} m; "
          f"set EE_STEP_M so the policy spans the standoff in ~30-60 steps (e.g. "
          f"{min(r[3] for r in results)/45:.3f}).")
    print("====================================\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
