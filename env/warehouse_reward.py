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


def _robot_xy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return robot xy position relative to env origin, shape (num_envs, 2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]


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
    threshold: float = 0.5,
) -> torch.Tensor:
    """+1 per step while robot xy is within `threshold` of current goal xy."""
    dist = torch.norm(_robot_xy(env, asset_cfg) - _current_goal_xy(env), dim=-1)
    return (dist < threshold).float()


def distance_to_goal(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Negative shaping reward: -distance(robot, goal). Encourages approach."""
    return -torch.norm(_robot_xy(env, asset_cfg) - _current_goal_xy(env), dim=-1)


def time_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Constant +1 per env per step. Combine with negative weight for time cost."""
    return torch.ones(env.num_envs, device=env.device)


def reached_goal(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.5,
) -> torch.Tensor:
    """Termination: True once robot xy is within threshold of goal xy."""
    dist = torch.norm(_robot_xy(env, asset_cfg) - _current_goal_xy(env), dim=-1)
    return dist < threshold


def out_of_bounds(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    half_extent_x: float = 9.5,
    half_extent_y: float = 14.5,
) -> torch.Tensor:
    """Termination: True if robot leaves the rectangular room interior."""
    xy = _robot_xy(env, asset_cfg)
    return (xy[:, 0].abs() > half_extent_x) | (xy[:, 1].abs() > half_extent_y)
