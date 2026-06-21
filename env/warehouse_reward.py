# warehouse_reward.py
# Person 1 — Reward and termination MDP functions for warehouse env.
#
# Signature matches isaaclab.managers.RewardTermCfg / TerminationTermCfg:
#     func(env: ManagerBasedRLEnv, ...) -> torch.Tensor[num_envs]

"""Reward and termination terms for the warehouse robot task."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def _robot_xy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return robot CHASSIS xy relative to env origin, shape (num_envs, 2).

    Reads body_pos_w["base_link"], NOT root_pos_w. The Ridgeback-Franka is a fixed-root
    articulation whose root_pos_w stays at the spawn pose while the chassis moves via the dummy
    base joints; reading the root froze every distance/termination at spawn (the robot never
    appeared to approach the goal, so shaping/success/out-of-bounds were all wrong). See
    IsaacLab issue #1268.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    idx = asset.body_names.index("base_link")  # moving chassis body (NOT the fixed root)
    return asset.data.body_pos_w[:, idx, :2] - env.scene.env_origins[:, :2]


def _current_goal_xy(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Read per-env goal xy from env.goal_pos (set by WarehouseRLEnv._reset_idx).

    Falls back to origin if attribute missing (early init / non-warehouse subclass).
    """
    if hasattr(env, "goal_pos"):
        return env.goal_pos[:, :2]
    return torch.zeros(env.num_envs, 2, device=env.device)


def delivery_success(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 1.5,
) -> torch.Tensor:
    """+1 per step while robot xy is within `threshold` of current goal xy.

    threshold=1.5m matches zone edge (3x3m zone → ±1.5m from center).
    """
    dist = torch.norm(_robot_xy(env, asset_cfg) - _current_goal_xy(env), dim=-1)
    return (dist < threshold).float()


def distance_to_goal(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Positive distance(robot, goal). Use with negative weight in RewardsCfg for shaping."""
    return torch.norm(_robot_xy(env, asset_cfg) - _current_goal_xy(env), dim=-1)


def time_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Constant +1 per env per step. Combine with negative weight for time cost."""
    return torch.ones(env.num_envs, device=env.device)


def reached_goal(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 1.5,
) -> torch.Tensor:
    """Termination: True once robot xy is within threshold of goal xy."""
    dist = torch.norm(_robot_xy(env, asset_cfg) - _current_goal_xy(env), dim=-1)
    return dist < threshold


def collision_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
    threshold_n: float = 5.0,
) -> torch.Tensor:
    """Returns -1.0 for envs with net contact force above threshold_n Newtons, else 0.

    Use with positive weight in RewardsCfg (e.g. weight=5.0 → effective penalty=-5 per step).
    Requires ContactSensorCfg on Robot chassis (see warehouse_scene.py).
    """
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    # net_forces_w_history is (N, T, B, 3): envs, history, bodies, xyz. Take latest frame (T=0),
    # force magnitude per body, then MAX over bodies → (N,). Without amax the body dim leaks as
    # (N, B)=[1,1] and reward_manager's `reward_buf += value` raises a [1] vs [1,1] broadcast error.
    net_force = sensor.data.net_forces_w_history[:, 0, :, :].norm(dim=-1).amax(dim=-1)
    return -(net_force > threshold_n).float()


def idle_penalty(
    env: ManagerBasedRLEnv,
    idle_steps: int = 50,
) -> torch.Tensor:
    """Returns -1.0 for envs idle (no base translation) for >= idle_steps consecutive steps, else 0.

    Reads env._stuck_steps (the idle counter maintained by WarehouseRLEnv._update_stuck). Use with
    a positive weight (e.g. 0.02 → -0.02/step) to make FREEZING strictly more expensive than careful
    movement — breaks the "diam aja to dodge the collision penalty" trap. Fires far earlier than the
    stuck_timeout reset (idle_steps=50 ≈ 5s vs STUCK_STEPS=450 ≈ 45s) so the policy feels the cost of
    standing still long before the episode is aborted.
    """
    if hasattr(env, "_stuck_steps"):
        return -(env._stuck_steps >= idle_steps).float()
    return torch.zeros(env.num_envs, device=env.device)


def out_of_bounds(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    half_extent_x: float = 9.5,
    half_extent_y: float = 14.5,
) -> torch.Tensor:
    """Termination: True if robot leaves the rectangular room interior."""
    xy = _robot_xy(env, asset_cfg)
    return (xy[:, 0].abs() > half_extent_x) | (xy[:, 1].abs() > half_extent_y)
