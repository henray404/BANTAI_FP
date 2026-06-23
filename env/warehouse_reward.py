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
    """Termination: True if robot leaves the rectangular room interior.

    Grace-gated: a step-0 contact transient can fling the chassis out before the spawn settles, so
    bounds is suppressed for the first RESET_GRACE_STEPS (spawn is always in-bounds by config).
    """
    xy = _robot_xy(env, asset_cfg)
    oob = (xy[:, 0].abs() > half_extent_x) | (xy[:, 1].abs() > half_extent_y)
    return oob & _past_grace(env)


# ── Failure resets + penalties (2026-06-23) ───────────────────────────
# A real CRASH (reset) uses a higher force than the soft per-step collision penalty: a brush (5N)
# is penalised but tolerated; a crash (>= COLLIDE_RESET_N) ends the episode.
COLLIDE_RESET_N = 50.0
# Rack footprint half-extents (m) for "under/inside a rack" detection. RACK_COLLIDER_SIZE is
# ~1.2x0.9; halves 0.6x0.45 padded a little so grazing the frame counts.
RACK_HALF_X = 0.70
RACK_HALF_Y = 0.55
NO_GRASP_TIMEOUT_STEPS = 300   # 30 s @ 10 Hz with no grasp -> reset (arm-first focus)
# Spawn-settle grace: the teleport/reset's first physics steps spike chassis contact (a few kN) as
# PhysX resolves initial floor/penetration — that tripped crashed/bounds at ep_len 0 (DIAG 2026-06-23).
# Suppress the failure terminations for the first RESET_GRACE_STEPS so the spawn settles first.
RESET_GRACE_STEPS = 5


def _contact_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Max net contact-force magnitude on the chassis bodies, shape (num_envs,)."""
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    return sensor.data.net_forces_w_history[:, 0, :, :].norm(dim=-1).amax(dim=-1)


def _past_grace(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(N,) bool: episode past the spawn-settle window — gate failure terminations with this so a
    step-0 teleport/contact transient can't end the episode. All-True if episode_length_buf absent."""
    n = getattr(env, "episode_length_buf", None)
    if n is None:
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    return n >= RESET_GRACE_STEPS


def collided(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
    threshold_n: float = COLLIDE_RESET_N,
) -> torch.Tensor:
    """Termination: chassis contact force above the CRASH threshold (auto-reset on nabrak).

    Suppressed during the spawn-settle grace window (RESET_GRACE_STEPS) so the reset teleport's
    initial contact transient doesn't false-trip it at ep_len 0.
    """
    return (_contact_force(env, sensor_cfg) > threshold_n) & _past_grace(env)


def _rack_xy(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Cached (R,2) tensor of env-local rack centres — only the SPAWNED racks (ACTIVE_RACK_POSITIONS
    honours scene.num_boxes), so under-rack never penalises an empty position."""
    if not hasattr(env, "_rack_xy_buf"):
        try:
            from env.warehouse_scene import ACTIVE_RACK_POSITIONS as _racks
        except Exception:
            from env.layout_grid import RACK_POSITIONS as _racks
        env._rack_xy_buf = torch.tensor(
            [[x, y] for (x, y, _z) in _racks], device=env.device, dtype=torch.float32
        )
    return env._rack_xy_buf


def under_rack(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    half_x: float = RACK_HALF_X,
    half_y: float = RACK_HALF_Y,
) -> torch.Tensor:
    """(N,) bool: robot chassis xy inside ANY rack footprint (drove under/into a rack)."""
    xy = _robot_xy(env, asset_cfg)                       # (N,2) env-local
    racks = _rack_xy(env)                                # (R,2) env-local
    dx = (xy[:, None, 0] - racks[None, :, 0]).abs()
    dy = (xy[:, None, 1] - racks[None, :, 1]).abs()
    return ((dx < half_x) & (dy < half_y)).any(dim=-1)   # (N,)


def under_rack_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    half_x: float = RACK_HALF_X,
    half_y: float = RACK_HALF_Y,
) -> torch.Tensor:
    """-1 per step while under a rack (use with POSITIVE weight, like collision_penalty).

    Explicit params (no **kwargs) — Isaac's manager introspects the signature and rejects *kw.
    """
    return -under_rack(env, asset_cfg=asset_cfg, half_x=half_x, half_y=half_y).float()


def no_grasp_timeout(
    env: ManagerBasedRLEnv,
    timeout_steps: int = NO_GRASP_TIMEOUT_STEPS,
) -> torch.Tensor:
    """Termination: still no grasp after timeout_steps and not holding (forces the arm-first task).

    Skips envs that have ever grasped this episode (env._ever_grasped) so a robot that grasped then
    is mid-carry/redelivering is not aborted.
    """
    if not (hasattr(env, "holding") and hasattr(env, "episode_length_buf")):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    # Only the grasp-isolation stage (Stage 2, spawn-near-box) enforces this — full-chain stages
    # (3/4) need the time to NAVIGATE to the box first, so a 30s grasp deadline would reset them
    # prematurely. Stage 1 is pregrasped (ever_grasped=True) so it never fires there anyway.
    from env.curriculum import stage_is_spawn_near_box
    if not stage_is_spawn_near_box(getattr(env, "stage", 0)):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    ever = getattr(env, "_ever_grasped", None)
    never = (~ever) if ever is not None else torch.ones_like(env.holding)
    return (env.episode_length_buf >= timeout_steps) & (~env.holding) & never
