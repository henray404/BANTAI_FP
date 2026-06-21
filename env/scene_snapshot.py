# env/scene_snapshot.py
# P5 — capture / restore the FULL initial scene state for faithful GUI replay.
#
# Action-only replay diverges: the env randomizes box poses + goal each reset. To replay the
# best episode exactly, we snapshot the scene at episode start (robot pose+joints, every box
# pose, the commanded goal/target) and restore it before applying the recorded actions.
#
# Operates on a WarehouseRLEnv (num_envs == 1). capture returns a JSON-serializable dict
# (plain lists) for best_init.json; restore writes it back into the sim.
#
# IMPORT RULE: no AppLauncher here (the replay entry owns it). torch + the live env only.
# UNVERIFIED on hardware (Blackwell camera blocker) — verify on the Linux box.

"""Capture + restore the warehouse scene's initial state for deterministic replay."""

from __future__ import annotations

import torch

from env.warehouse_scene import ITEM_SPECS


def _box_names() -> list[str]:
    """Names of the graspable boxes whose poses must be snapshotted."""
    return [spec[0] for spec in ITEM_SPECS]


def read_replay_state(rl_env) -> tuple[list[float], list[float], float]:
    """Return env-0 (robot_base_xyz, ee_xyz, holding) for a trajectory CSV row."""
    robot = rl_env.scene["robot"]
    idx = robot.body_names.index("base_link")
    base = (robot.data.body_pos_w[0, idx] - rl_env.scene.env_origins[0])
    return (base.detach().cpu().tolist(),
            rl_env.ee_pos[0].detach().cpu().tolist(),
            float(rl_env.holding[0].item()))


def capture_init_state(rl_env) -> dict:
    """Snapshot env-0 scene state to a JSON-serializable dict.

    Captures: robot root state + joint pos/vel, every box's root state, and the commanded
    goal/target buffers. Restore with restore_init_state() before replaying actions.
    """
    robot = rl_env.scene["robot"]
    snap: dict = {
        "robot_root": robot.data.root_state_w[0].detach().cpu().tolist(),      # (13,)
        "robot_joint_pos": robot.data.joint_pos[0].detach().cpu().tolist(),
        "robot_joint_vel": robot.data.joint_vel[0].detach().cpu().tolist(),
        "boxes": {n: rl_env.scene[n].data.root_state_w[0].detach().cpu().tolist()
                  for n in _box_names()},
        "goal_id": rl_env.goal_id_buf[0].detach().cpu().tolist(),
        "box_cat_idx": int(rl_env.box_cat_idx[0].item()),
        "target_box_name": rl_env.target_box_name[0],
        "goal_pos": rl_env.goal_pos[0].detach().cpu().tolist(),
        "box_pos": rl_env.box_pos[0].detach().cpu().tolist(),
    }
    return snap


def restore_init_state(rl_env, snap: dict) -> None:
    """Write a captured snapshot back into env 0 (call right after env.reset())."""
    dev = rl_env.device
    eid = torch.tensor([0], device=dev)

    def _t(x):
        return torch.tensor(x, device=dev, dtype=torch.float32).unsqueeze(0)

    robot = rl_env.scene["robot"]
    robot.write_root_state_to_sim(_t(snap["robot_root"]), env_ids=eid)
    robot.write_joint_state_to_sim(_t(snap["robot_joint_pos"]),
                                   _t(snap["robot_joint_vel"]), env_ids=eid)
    for name, state in snap["boxes"].items():
        rl_env.scene[name].write_root_state_to_sim(_t(state), env_ids=eid)

    # Commanded goal/target buffers (so reward/obs/holding match the recorded episode).
    rl_env.goal_id_buf[0] = torch.tensor(snap["goal_id"], device=dev, dtype=torch.float32)
    rl_env.box_cat_idx[0] = int(snap["box_cat_idx"])
    rl_env.target_box_name[0] = snap["target_box_name"]
    rl_env.goal_pos[0] = torch.tensor(snap["goal_pos"], device=dev, dtype=torch.float32)
    rl_env.box_pos[0] = torch.tensor(snap["box_pos"], device=dev, dtype=torch.float32)
    rl_env.holding[0] = False
