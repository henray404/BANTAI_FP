# recording/sim_state.py
# Person 5 — capture/restore the full sim state for checkpoint rewind (Isaac glue, import-safe).
#
# capture_sim_state(env) -> blob : grab robot joints + target box pose + task flags for env 0.
# restore_sim_state(env, blob)   : write them back into the sim (the "rewind").
# torch is imported inside the functions, so this module loads fine on a Mac.

"""Snapshot and restore the real WarehouseGymEnv state for checkpoint rewind."""

from __future__ import annotations

from typing import Any

from recording.state_extractor import _inner


def capture_sim_state(env) -> dict[str, Any]:
    """Snapshot env-0 robot joint state + target box root state + task flags."""
    ie = _inner(env)
    robot = ie.scene["robot"]
    box_name = ie.target_box_name[0]
    box = ie.scene[box_name]
    return {
        "joint_pos": robot.data.joint_pos[0:1].clone(),
        "joint_vel": robot.data.joint_vel[0:1].clone(),
        "box_root_state": box.data.root_state_w[0:1].clone(),
        "box_name": box_name,
        "holding": bool(ie.holding[0].item()),
    }


def restore_sim_state(env, blob: dict[str, Any]) -> None:
    """Write a captured snapshot back into the sim (rewind robot joints + box pose + flags)."""
    import torch

    ie = _inner(env)
    dev = ie.device
    ids = torch.tensor([0], device=dev)
    robot = ie.scene["robot"]

    robot.write_joint_state_to_sim(blob["joint_pos"].to(dev), blob["joint_vel"].to(dev), env_ids=ids)
    box = ie.scene[blob["box_name"]]
    state = blob["box_root_state"].to(dev).clone()
    state[:, 7:13] = 0.0  # zero velocities so the box doesn't fly off after the rewind
    box.write_root_state_to_sim(state, env_ids=ids)
    robot.write_data_to_sim()

    ie.holding[0] = bool(blob["holding"])
    ie._refresh_target_box_pos()  # keep env.box_pos consistent with the restored box pose
