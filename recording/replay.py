# recording/replay.py
# Person 5 — faithful replay: write the RECORDED state into the sim each step (no re-simulation).
#
# For a demo we must show the EXACT best run, so we don't re-feed actions and hope physics matches.
# Instead each step we set the robot's joint positions/velocities and the target box pose to the
# recorded values, then render. The kinematics are reproduced bit-for-bit from the CSV.
#
# IMPORT-SAFE: torch is imported inside the functions, so this module loads on a Mac.

"""Apply recorded joint + box state back into a live WarehouseGymEnv for faithful playback."""

from __future__ import annotations

from typing import Any

from recording.recorder import TrajectoryReader
from recording.state_extractor import _inner


def apply_row_to_env(env, row: dict[str, Any], joint_names: list[str]) -> None:
    """Write one recorded step's robot joints + target box pose into the sim (kinematic playback)."""
    import torch

    ie = _inner(env)
    dev = ie.device
    robot = ie.scene["robot"]

    q = torch.tensor([[float(row[f"q_{n}"]) for n in joint_names]], dtype=torch.float32, device=dev)
    qd = torch.tensor([[float(row.get(f"qd_{n}", 0.0)) for n in joint_names]],
                      dtype=torch.float32, device=dev)
    robot.write_joint_state_to_sim(q, qd, env_ids=torch.tensor([0], device=dev))

    # Target box: pos (xyz) + quat (wxyz), zero velocity.
    box_name = ie.target_box_name[0]
    box = ie.scene[box_name]
    state = box.data.root_state_w[0:1].clone()
    state[0, 0:3] = torch.tensor([row["box_x"], row["box_y"], row["box_z"]], device=dev)
    state[0, 3:7] = torch.tensor([row["box_qw"], row["box_qx"], row["box_qy"], row["box_qz"]], device=dev)
    state[0, 7:13] = 0.0
    box.write_root_state_to_sim(state, env_ids=torch.tensor([0], device=dev))

    robot.write_data_to_sim()


def reset_to_recorded_scenario(env, reader: TrajectoryReader) -> None:
    """Reset the env to the recorded scenario via the saved seed (reproduces box/goal/spawn draw).

    Replay overwrites poses every step anyway, but seeding first lines up the static scene (which
    box is the target, zone, spawn region) with the recording.
    """
    seed = reader.meta.get("seed")
    if seed is not None:
        env.reset(seed=int(seed))
    else:
        env.reset()


def replay(env, path, simulation_app=None, sleep_dt: bool = True) -> None:
    """Replay a recorded run faithfully: seed scene, then set recorded state each step and render.

    Args:
        env:            a live WarehouseGymEnv (num_envs=1).
        path:           run path (.csv / .meta.json / stem).
        simulation_app: Isaac SimulationApp; loop stops when it stops running.
        sleep_dt:       pace playback at the recorded control_dt for a watchable demo.
    """
    import time

    reader = TrajectoryReader(path)
    reset_to_recorded_scenario(env, reader)
    dt = float(reader.meta.get("control_dt", 0.1))
    ie = _inner(env)

    for row in reader.rows:
        if simulation_app is not None and not simulation_app.is_running():
            break
        apply_row_to_env(env, row, reader.joint_names)
        ie.sim.render() if hasattr(ie, "sim") else None
        if sleep_dt:
            time.sleep(dt)
