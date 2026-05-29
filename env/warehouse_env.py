# warehouse_env.py
# Person 1 — Environment & Integration
#
# IMPORT RULE: No AppLauncher here. Imported by tests/test_env.py and scripts/run_env.py
# which own AppLauncher. See bugs_errors/2026-05-15_double-applaunch-crash.md.
#
# Interface contract (DO NOT CHANGE without team discussion):
#   obs = {
#       "pixels":   Tensor(batch, 3, 64, 64),   # camera image, float in [0,1]
#       "position": Tensor(batch, 3),            # robot xyz, env-local
#       "goal":     Tensor(batch, 3),            # target zone xyz (curriculum: anneal→zeros later)
#       "goal_emb": Tensor(batch, 512),          # CLIP embedding placeholder (zeros until P4)
#   }
#   action_space = Box(-1, 1, shape=(2,))       # [linear_vel, angular_vel]
#
# NOTE: velocity obs deliberately EXCLUDED. Council + TEEP consensus: DreamerV3 RSSM
# reconstructs motion from pixel sequences — adding velocity is redundant + breaks P2 contract.

"""Warehouse environment: ManagerBasedRLEnvCfg + Gymnasium wrapper for DreamerV3."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

import isaaclab.envs.mdp as mdp
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import TiledCamera
from isaaclab.utils import configclass

from env.warehouse_scene import ZONE_SPECS, WarehouseSceneCfg
from env.warehouse_reward import (
    delivery_success,
    distance_to_goal,
    out_of_bounds,
    reached_goal,
    time_penalty,
)


# ── Constants ─────────────────────────────────────────────────────────
GOAL_EMB_DIM  = 512   # CLIP embedding size (zeros until Person 4 wires CLIP)
IMG_HW        = 64    # camera resolution (square)
WHEEL_BASE    = 0.118 # Jetbot wheel separation in meters
WHEEL_RADIUS  = 0.032 # Jetbot wheel radius in meters
MAX_LIN_SPEED = 1.0   # max linear speed m/s (DreamerNav standard; was wrongly 0.32 m/s)
MAX_ANG_SPEED = 2.0   # max angular speed rad/s


# ── Custom Observation Functions ──────────────────────────────────────
def camera_rgb(env: ManagerBasedRLEnv, sensor_name: str = "camera") -> torch.Tensor:
    """Return RGB image tensor (num_envs, 3, H, W) in float [0, 1]."""
    cam: TiledCamera = env.scene[sensor_name]
    rgb: torch.Tensor = cam.data.output["rgb"]
    if rgb.dtype == torch.uint8:
        rgb = rgb.float() / 255.0
    elif rgb.max() > 1.5:
        rgb = rgb / 255.0
    return rgb[..., :3].permute(0, 3, 1, 2).contiguous()


def robot_position(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return robot xyz relative to env origin, shape (num_envs, 3)."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.root_pos_w - env.scene.env_origins


def goal_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return current per-env goal xyz, shape (num_envs, 3)."""
    if hasattr(env, "goal_pos"):
        return env.goal_pos
    return torch.zeros(env.num_envs, 3, device=env.device)


def goal_embedding(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return placeholder goal embedding (zeros). Filled by Person 4 with CLIP."""
    return torch.zeros(env.num_envs, GOAL_EMB_DIM, device=env.device)


# ── Actions ───────────────────────────────────────────────────────────
@configclass
class ActionsCfg:
    """Wheel velocity control. Action dim = 2 ([left_wheel, right_wheel] rad/s)."""

    # scale=1.0 — _diff_drive in WarehouseGymEnv converts [lin,ang]→[L,R] rad/s directly.
    wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
    )


# ── Observations ──────────────────────────────────────────────────────
@configclass
class ObservationsCfg:
    """Returns dict obs matching interface contract (no concatenation)."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy obs group: pixels, position, goal, goal_emb."""

        pixels   = ObsTerm(func=camera_rgb)
        position = ObsTerm(func=robot_position)
        goal     = ObsTerm(func=goal_position)
        goal_emb = ObsTerm(func=goal_embedding)

        def __post_init__(self) -> None:
            """Keep terms as separate keys instead of concatenating."""
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


# ── Events ────────────────────────────────────────────────────────────
@configclass
class EventCfg:
    """Reset robot pose at episode start."""

    reset_robot = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            # Receiving area (north); robot must navigate south through islands to a zone.
            "pose_range": {"x": (-8.0, 8.0), "y": (11.0, 14.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    )


# ── Rewards ───────────────────────────────────────────────────────────
@configclass
class RewardsCfg:
    """Reward terms.

    success:  +1 per step while at goal (sparse, dominant)
    shaping:  -dist(robot, goal) per step (dense, small weight)
    time:     -0.001 per step (efficiency)
    """

    success  = RewTerm(func=delivery_success, weight=1.0)
    shaping  = RewTerm(func=distance_to_goal, weight=0.05)
    time_pen = RewTerm(func=time_penalty,     weight=-0.001)


# ── Terminations ──────────────────────────────────────────────────────
@configclass
class TerminationsCfg:
    """Episode end conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success  = DoneTerm(func=reached_goal)
    bounds   = DoneTerm(func=out_of_bounds, params={"half_extent_x": 9.5, "half_extent_y": 14.5})


# ── Env Cfg ───────────────────────────────────────────────────────────
@configclass
class WarehouseEnvCfg(ManagerBasedRLEnvCfg):
    """Warehouse env config wiring scene + MDP managers together."""

    scene: WarehouseSceneCfg = WarehouseSceneCfg(num_envs=2, env_spacing=32.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Set sim + episode parameters.

        Physics: dt=0.005s (200 Hz). decimation=20 → policy at 10 Hz (DreamerNav standard).
        Episode 45s × 10Hz = 450 steps. Room 20×20m, max speed 1 m/s, ~13m to zone → ~13s
        minimum, leaving ~32s buffer for navigation around racks.
        """
        self.decimation = 20           # 200 Hz / 20 = 10 Hz control (DreamerNav std, saves VRAM)
        self.episode_length_s = 60.0   # 60s x 10Hz = 600 steps; ~25m traverse @1m/s + island nav
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.viewer.eye = (0.0, -20.0, 18.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)


# ── Custom RL Env ─────────────────────────────────────────────────────
class WarehouseRLEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv with a per-env `goal_pos` buffer sampled on reset."""

    def __init__(self, cfg: WarehouseEnvCfg, render_mode: str | None = None, **kwargs):
        """Init env, allocate goal buffer, and validate scene entities."""
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        self._validate()
        self.goal_pos: torch.Tensor = torch.zeros(self.num_envs, 3, device=self.device)
        # Candidate goals: env-local xyz of each zone (from ZONE_SPECS)
        self._zone_pos = torch.tensor(
            [pos for _, _, pos in ZONE_SPECS], device=self.device, dtype=torch.float32
        )
        self._resample_goals(torch.arange(self.num_envs, device=self.device))

    def _validate(self) -> None:
        """Assert critical scene entities exist and are correctly configured."""
        # Camera must be present (TiledCamera; CameraCfg crashes on Blackwell).
        if "camera" not in self.scene.sensors:
            raise RuntimeError(
                "Scene missing 'camera' sensor. "
                "Ensure TiledCameraCfg is defined in WarehouseSceneCfg. "
                "CameraCfg is NOT supported on RTX 5050 (Blackwell) — see "
                "bugs_errors/2026-05-22_sdp-camera-crash-blackwell.md"
            )
        # Robot must have exactly 2 wheel joints.
        robot: Articulation = self.scene["robot"]
        n_joints = robot.num_joints
        if n_joints != 2:
            raise RuntimeError(
                f"Expected 2 wheel joints on robot, found {n_joints}. "
                "Check joint_names_expr in ActionsCfg or verify Jetbot USD loaded correctly. "
                "If Nucleus is unreachable, the USD will fail silently with 0 joints."
            )

    def _resample_goals(self, env_ids: torch.Tensor) -> None:
        """Pick a random zone xyz for each env in `env_ids`."""
        if env_ids.numel() == 0:
            return
        idx = torch.randint(0, self._zone_pos.shape[0], (env_ids.numel(),), device=self.device)
        self.goal_pos[env_ids] = self._zone_pos[idx]

    def _reset_idx(self, env_ids) -> None:
        """Resample goals first so obs manager sees the new goal."""
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._resample_goals(env_ids_t)
        super()._reset_idx(env_ids)


# ── Gymnasium Wrapper ─────────────────────────────────────────────────
class WarehouseGymEnv(gym.Env):
    """Gymnasium-style wrapper around `WarehouseRLEnv`.

    Exposes:
        action_space      = Box(-1, 1, shape=(2,))    [linear_vel, angular_vel]
        observation_space = Dict(pixels, position, goal, goal_emb)

    Internally converts [linear, angular] -> [left_wheel, right_wheel] wheel rates
    before forwarding to the underlying ManagerBasedRLEnv. Returns batched
    tensors (num_envs, ...); single-env consumers should set num_envs=1.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg: WarehouseEnvCfg | None = None, render_mode: str | None = None):
        """Build underlying RL env and Gym-style spaces."""
        self.cfg = cfg if cfg is not None else WarehouseEnvCfg()
        self._env = WarehouseRLEnv(cfg=self.cfg, render_mode=render_mode)
        self.num_envs: int = self._env.num_envs
        self.device = self._env.device

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "pixels":   spaces.Box(0.0, 1.0, shape=(3, IMG_HW, IMG_HW), dtype=np.float32),
                "position": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "goal":     spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "goal_emb": spaces.Box(-np.inf, np.inf, shape=(GOAL_EMB_DIM,), dtype=np.float32),
            }
        )

    def _diff_drive(self, action: torch.Tensor) -> torch.Tensor:
        """Map [linear_vel, angular_vel] in [-1,1] → [left, right] wheel velocities in rad/s.

        ActionsCfg.scale=1.0, so values passed directly to joint velocity targets (rad/s).
        action[:,0] scaled by MAX_LIN_SPEED; action[:,1] scaled by MAX_ANG_SPEED.
        """
        lin_mps = action[:, 0] * MAX_LIN_SPEED    # m/s
        ang_rps = action[:, 1] * MAX_ANG_SPEED    # rad/s
        v_left  = (lin_mps - 0.5 * WHEEL_BASE * ang_rps) / WHEEL_RADIUS   # rad/s
        v_right = (lin_mps + 0.5 * WHEEL_BASE * ang_rps) / WHEEL_RADIUS   # rad/s
        return torch.stack([v_left, v_right], dim=-1)

    def _unwrap_obs(self, obs: dict) -> dict[str, torch.Tensor]:
        """Pull terms out of obs['policy'] dict to match interface contract."""
        policy = obs["policy"]
        if not isinstance(policy, dict):
            raise RuntimeError(
                "ObservationsCfg returned non-dict 'policy'. "
                "Ensure PolicyCfg.concatenate_terms = False."
            )
        return {
            "pixels":   policy["pixels"],
            "position": policy["position"],
            "goal":     policy["goal"],
            "goal_emb": policy["goal_emb"],
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reset all sub-envs; return (obs_dict, info)."""
        if seed is not None:
            torch.manual_seed(seed)
        obs, info = self._env.reset()
        return self._unwrap_obs(obs), info

    def step(self, action):
        """Apply [linear, angular] action; return (obs, reward, terminated, truncated, info)."""
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(self.device, dtype=torch.float32)
        if action.ndim == 1:
            action = action.unsqueeze(0).expand(self.num_envs, -1)
        action = action.clamp(-1.0, 1.0).to(self.device, dtype=torch.float32)
        wheel_action = self._diff_drive(action)
        obs, reward, terminated, truncated, info = self._env.step(wheel_action)
        return self._unwrap_obs(obs), reward, terminated, truncated, info

    def render(self):
        """Return env-0 camera RGB (uint8 H,W,3) for visualization."""
        cam: TiledCamera = self._env.scene["camera"]
        rgb = cam.data.output["rgb"][0]
        if rgb.dtype != torch.uint8:
            rgb = (rgb.clamp(0, 1) * 255).to(torch.uint8) if rgb.max() <= 1.5 else rgb.to(torch.uint8)
        return rgb[..., :3].cpu().numpy()

    def close(self) -> None:
        """Close underlying env."""
        self._env.close()
