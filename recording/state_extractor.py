# recording/state_extractor.py
# Person 5 — pull the COMPLETE per-step state out of the real WarehouseGymEnv for recording.
#
# IMPORT-SAFE: no torch/Isaac import at module load. Tensors are read duck-typed (.tolist()/float()),
# so this file imports fine on a Mac; the actual reads only run inside the Isaac process.
#
# "Record everything" = for env 0 each step we capture:
#   - ALL joint positions + velocities (base dummy x/y/z + Franka 7 + gripper 2) — the kinematics
#   - base_link world pose (xyz + quaternion + roll/pitch/yaw) — the moving chassis (NOT root)
#   - end-effector world pose + base-frame ee_pos
#   - target box world pose (xyz + quaternion) + env-local box_pos
#   - gripper opening, holding, grasp/drop/deliver events
#   - goal zone xyz + goal_id one-hot + category/color
#   - action (6), reward, terminated/truncated/success, chassis contact force
# Plus build_metadata() captures the run-once scenario + env config + joint name order for replay.

"""Extract the full robot+scene state from WarehouseGymEnv into flat record rows + run metadata."""

from __future__ import annotations

import math
from typing import Any

CATEGORY_NAMES = ("fragile", "regular", "heavy")
CATEGORY_COLORS = ("orange", "cyan", "purple")


def _f(x) -> float:
    """Scalar tensor/number → float."""
    try:
        return float(x.item())
    except AttributeError:
        return float(x)


def _l(x) -> list[float]:
    """1-D tensor/array → list[float]."""
    return [float(v) for v in x.tolist()]


def _rpy_deg(quat_wxyz) -> tuple[float, float, float]:
    """(w,x,y,z) quaternion → (roll, pitch, yaw) in degrees."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    roll = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))
    yaw = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return roll, pitch, yaw


def _inner(env):
    """The WarehouseRLEnv that owns the buffers (handles WarehouseGymEnv or the RL env directly)."""
    return getattr(env, "_env", env)


def joint_names(env) -> list[str]:
    """Ordered joint names of the robot articulation (column order for q_/qd_)."""
    return list(_inner(env).scene["robot"].joint_names)


def build_metadata(env, *, seed: int | None, policy: str, run_id: str, extra: dict | None = None) -> dict:
    """Capture the run-once scenario + env config needed to reconstruct and replay this run.

    Recorded for env 0: seed, category/color/goal_id, target box id + size, spawn base pose, zone
    center, env origin, control dt / episode length, joint name order, gripper open width.
    """
    ie = _inner(env)
    robot = ie.scene["robot"]
    blink = robot.body_names.index("base_link")
    cat = int(ie.box_cat_idx[0].item()) if hasattr(ie, "box_cat_idx") else -1
    box_name = ie.target_box_name[0] if hasattr(ie, "target_box_name") else None
    meta = {
        "run_id": run_id,
        "policy": policy,
        "seed": seed,
        "num_envs": int(getattr(ie, "num_envs", 1)),
        "control_dt": _f(getattr(ie, "step_dt", 0.1)),
        "control_hz": round(1.0 / _f(getattr(ie, "step_dt", 0.1)), 3),
        "max_episode_steps": int(getattr(ie, "max_episode_length", 0)),
        "category_idx": cat,
        "category": CATEGORY_NAMES[cat] if 0 <= cat < 3 else "unknown",
        "color": CATEGORY_COLORS[cat] if 0 <= cat < 3 else "unknown",
        "goal_id": _l(ie.goal_id_buf[0]) if hasattr(ie, "goal_id_buf") else None,
        "target_box_name": box_name,
        "box_size_m": _f(ie._box_size[box_name]) if box_name and hasattr(ie, "_box_size") else None,
        "goal_zone_xyz": _l(ie.goal_pos[0]) if hasattr(ie, "goal_pos") else None,
        "env_origin": _l(ie.scene.env_origins[0]),
        "spawn_base_pose_w": _l(robot.data.body_pos_w[0, blink]) + _l(robot.data.body_quat_w[0, blink]),
        "joint_names": joint_names(env),
        "action_layout": ["base_lin", "base_ang", "ee_dx", "ee_dy", "ee_dz", "gripper"],
    }
    if extra:
        meta.update(extra)
    return meta


def step_row(env, *, step: int, t: float, action, reward, terminated, truncated, info: dict | None,
             slope_reward: float | None = None) -> dict[str, Any]:
    """Build one flat record row capturing the complete env-0 state after a step.

    `action` is the (6,) command applied this step; `reward` the scalar env reward; `info` the
    step info (grasp/deliver/drop events if the env/wrapper provided them).
    """
    ie = _inner(env)
    robot = ie.scene["robot"]
    names = robot.joint_names
    jp = robot.data.joint_pos[0]
    jv = robot.data.joint_vel[0]

    blink = robot.body_names.index("base_link")
    hand = robot.body_names.index("panda_hand")
    base_p = robot.data.body_pos_w[0, blink]
    base_q = robot.data.body_quat_w[0, blink]
    ee_p_w = robot.data.body_pos_w[0, hand]
    ee_q_w = robot.data.body_quat_w[0, hand]
    roll, pitch, yaw = _rpy_deg(base_q)

    box_name = ie.target_box_name[0]
    box = ie.scene[box_name]
    box_p = box.data.root_pos_w[0]
    box_q = box.data.root_quat_w[0]

    a = _l(action[0]) if hasattr(action, "__len__") and not isinstance(action, (int, float)) else _l(action)
    info = info or {}

    row: dict[str, Any] = {
        "step": step,
        "t": round(float(t), 4),
        # action (6)
        "a_base_lin": round(a[0], 5), "a_base_ang": round(a[1], 5),
        "a_ee_dx": round(a[2], 5), "a_ee_dy": round(a[3], 5), "a_ee_dz": round(a[4], 5),
        "a_grip": round(a[5], 5),
        # base_link world pose (moving chassis)
        "base_x": round(_f(base_p[0]), 5), "base_y": round(_f(base_p[1]), 5), "base_z": round(_f(base_p[2]), 5),
        "base_qw": round(_f(base_q[0]), 6), "base_qx": round(_f(base_q[1]), 6),
        "base_qy": round(_f(base_q[2]), 6), "base_qz": round(_f(base_q[3]), 6),
        "base_roll_deg": round(roll, 3), "base_pitch_deg": round(pitch, 3), "base_yaw_deg": round(yaw, 3),
        # end-effector
        "ee_x": round(_f(ee_p_w[0]), 5), "ee_y": round(_f(ee_p_w[1]), 5), "ee_z": round(_f(ee_p_w[2]), 5),
        "ee_qw": round(_f(ee_q_w[0]), 6), "ee_qx": round(_f(ee_q_w[1]), 6),
        "ee_qy": round(_f(ee_q_w[2]), 6), "ee_qz": round(_f(ee_q_w[3]), 6),
        "ee_base_x": round(_f(ie.ee_pos[0][0]), 5), "ee_base_y": round(_f(ie.ee_pos[0][1]), 5),
        "ee_base_z": round(_f(ie.ee_pos[0][2]), 5),
        # target box world pose
        "box_x": round(_f(box_p[0]), 5), "box_y": round(_f(box_p[1]), 5), "box_z": round(_f(box_p[2]), 5),
        "box_qw": round(_f(box_q[0]), 6), "box_qx": round(_f(box_q[1]), 6),
        "box_qy": round(_f(box_q[2]), 6), "box_qz": round(_f(box_q[3]), 6),
        # grasp / goal state
        "gripper": round(_f(ie.scene["robot"].data.joint_pos[0, names.index("panda_finger_joint1")]) / 0.035, 4),
        "holding": int(_f(ie.holding[0])),
        "grasp_event": int(bool(info.get("grasp_event", _f(getattr(ie, "grasp_event", [0])[0]) if hasattr(ie, "grasp_event") else 0))),
        "drop_event": int(bool(info.get("drop_event", _f(getattr(ie, "drop_event", [0])[0]) if hasattr(ie, "drop_event") else 0))),
        "goal_x": round(_f(ie.goal_pos[0][0]), 5), "goal_y": round(_f(ie.goal_pos[0][1]), 5),
        "goal_z": round(_f(ie.goal_pos[0][2]), 5),
        # reward / termination
        "reward": round(_f(reward[0]) if hasattr(reward, "__len__") else _f(reward), 5),
        "slope_reward": round(float(slope_reward), 5) if slope_reward is not None else "",
        "terminated": int(_f(terminated[0]) if hasattr(terminated, "__len__") else _f(terminated)),
        "truncated": int(_f(truncated[0]) if hasattr(truncated, "__len__") else _f(truncated)),
        "contact_force_n": round(_contact_force(ie), 4),
    }
    # ALL joints last (dynamic columns q_<name> / qd_<name>) — the kinematics, never dropped.
    for i, n in enumerate(names):
        row[f"q_{n}"] = round(_f(jp[i]), 6)
    for i, n in enumerate(names):
        row[f"qd_{n}"] = round(_f(jv[i]), 6)
    return row


def _contact_force(ie) -> float:
    """Max chassis contact-force magnitude (N), 0.0 if no contact sensor present."""
    try:
        sensor = ie.scene["contact_sensor"]
        net = sensor.data.net_forces_w_history[0, 0]  # (bodies, 3)
        return max(float((row_[0] ** 2 + row_[1] ** 2 + row_[2] ** 2) ** 0.5) for row_ in net.tolist())
    except Exception:
        return 0.0
