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

from env.warehouse_scene import ITEM_SPECS, SHELF_DECK_SIZE, ZONE_SPECS, WarehouseSceneCfg
from env.warehouse_reward import (
    collision_penalty,
    delivery_success,
    distance_to_goal,
    out_of_bounds,
    reached_goal,
    time_penalty,
)


# ── Constants ─────────────────────────────────────────────────────────
GOAL_EMB_DIM  = 512   # CLIP embedding size (zeros until Person 4 wires CLIP)
IMG_HW        = 64    # camera resolution (square)
# Ridgeback-Franka holonomic base: no wheels — driven via dummy base joints (velocity ctrl).
# _base_cmd maps the (2,) action [linear, angular] directly to base joint velocities; there is
# no wheel-radius/separation kinematics (that was for the old diff-drive Carter/Jetbot).
MAX_LIN_SPEED = 1.5   # max base linear speed m/s
MAX_ANG_SPEED = 1.5   # max base yaw rate rad/s


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


def robot_heading(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return [cos(yaw), sin(yaw)] heading, shape (num_envs, 2).

    Unit-circle encoding avoids the ±π discontinuity of raw yaw.
    Critical because robot spawns at random yaw — policy needs heading to orient itself.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    quat = robot.data.root_quat_w  # (w, x, y, z)
    yaw = torch.atan2(
        2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
        1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2),
    )
    return torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)


# ── Actions ───────────────────────────────────────────────────────────
@configclass
class ActionsCfg:
    """Holonomic base velocity control. Internal action dim = 3 (3 dummy base joints).

    The external policy action stays (2,) [linear, angular]; WarehouseGymEnv._base_cmd expands
    it to [vx, vy=0, wz] in this joint order. preserve_order keeps the column order matching the
    joint_names list so _base_cmd's output lines up. Arm + gripper are NOT actuated here (held by
    their position-control actuators); the pickup state machine drives them separately.
    """

    # scale=1.0 — _base_cmd emits joint velocity targets directly.
    base_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=[
            "dummy_base_prismatic_x_joint",
            "dummy_base_prismatic_y_joint",
            "dummy_base_revolute_z_joint",
        ],
        preserve_order=True,
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
        heading  = ObsTerm(func=robot_heading)   # [cos(yaw), sin(yaw)] — needed for random-spawn orientation

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

    success:   +10 per step while at goal (was +1; raised to balance shaping scale)
    shaping:   -0.01 * dist(robot, goal) per step (weight negative; func returns positive dist)
    time_pen:  -0.005 per step (slightly higher than before for efficiency pressure)
    collision: -5 per step when contact force > 5N on chassis (requires ContactSensor)
    """

    success   = RewTerm(func=delivery_success, weight=10.0)
    shaping   = RewTerm(func=distance_to_goal, weight=-0.01)   # sign in weight (func now returns +dist)
    time_pen  = RewTerm(func=time_penalty,     weight=-0.005)
    collision = RewTerm(func=collision_penalty, weight=5.0)    # func returns 0/-1; weight=5 → penalty=-5


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

    scene: WarehouseSceneCfg = WarehouseSceneCfg(num_envs=1, env_spacing=32.0)  # 1 env: Ridgeback-Franka + 54 rigid boxes saturates 8GB
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
        # Arm + contact stability (Ridgeback-Franka): reduce noisy base/arm velocities.
        self.sim.physx.enable_external_forces_every_iteration = True
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
        # Ridgeback-Franka: 3 base + 7 arm + 2 finger = 12 joints. The (2,) action drives the 3
        # dummy base joints; arm/gripper are held by position-control actuators.
        robot: Articulation = self.scene["robot"]
        n_joints = robot.num_joints
        if n_joints < 3:
            raise RuntimeError(
                f"Expected >=3 base joints on the Ridgeback-Franka robot, found {n_joints}. "
                "Check ActionsCfg.base_vel joint_names match the dummy_base_* joints, or verify "
                "ridgeback_franka.usd loaded (Nucleus unreachable -> USD fails silently, 0 joints)."
            )

    def _resample_goals(self, env_ids: torch.Tensor) -> None:
        """Pick a random zone xyz for each env in `env_ids`."""
        if env_ids.numel() == 0:
            return
        idx = torch.randint(0, self._zone_pos.shape[0], (env_ids.numel(),), device=self.device)
        self.goal_pos[env_ids] = self._zone_pos[idx]

    def _randomize_box_poses(self, env_ids: torch.Tensor) -> None:
        """Random x,y jitter for all 54 boxes within their shelf deck area.

        Called AFTER super()._reset_idx() so it overrides scene.reset() default positions.
        Each box stays on its shelf level (z fixed) but gets a new random x,y offset.
        Velocities zeroed to prevent carry-over from previous episode.
        """
        if env_ids.numel() == 0:
            return
        n = env_ids.numel()
        margin = 0.02  # safety gap from shelf edge (meters)

        for box_name, size, _mass, pos in ITEM_SPECS:
            box = self.scene[box_name]
            jlim_x = max(0.0, SHELF_DECK_SIZE[0] / 2.0 - size / 2.0 - margin)
            jlim_y = max(0.0, SHELF_DECK_SIZE[1] / 2.0 - size / 2.0 - margin)

            jx = (torch.rand(n, device=self.device) * 2.0 - 1.0) * jlim_x
            jy = (torch.rand(n, device=self.device) * 2.0 - 1.0) * jlim_y

            # root state: [px, py, pz, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz] world-frame
            state = torch.zeros(n, 13, device=self.device)
            state[:, 0] = self.scene.env_origins[env_ids, 0] + pos[0] + jx
            state[:, 1] = self.scene.env_origins[env_ids, 1] + pos[1] + jy
            state[:, 2] = self.scene.env_origins[env_ids, 2] + pos[2] + 0.05  # 5cm above deck
            state[:, 3] = 1.0  # qw — upright, no rotation
            box.write_root_state_to_sim(state, env_ids=env_ids)

    def _reset_idx(self, env_ids) -> None:
        """Resample goals, reset scene, then randomize box positions."""
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._resample_goals(env_ids_t)
        super()._reset_idx(env_ids)
        self._randomize_box_poses(env_ids_t)  # after super() so scene.reset() doesn't undo it


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
                "heading":  spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),  # [cos(yaw), sin(yaw)]
            }
        )

    def _base_cmd(self, action: torch.Tensor) -> torch.Tensor:
        """Map [linear, angular] in [-1,1] → holonomic base joint velocities [vx, vy, wz].

        Order matches ActionsCfg.base_vel joint_names (preserve_order=True):
        [dummy_base_prismatic_x, dummy_base_prismatic_y, dummy_base_revolute_z].
        prismatic_x = forward (linear), revolute_z = yaw (angular), prismatic_y = 0 (no strafe →
        diff-drive-like behaviour, keeps the (2,) contract).

        VERIFY on first drive: if the base translates in a FIXED WORLD direction regardless of
        heading, the dummy-joint chain is world-framed — then project: vx=lin*cos(yaw),
        vy=lin*sin(yaw) using the robot yaw, instead of [lin, 0].
        """
        lin = action[:, 0] * MAX_LIN_SPEED    # m/s along base x
        ang = action[:, 1] * MAX_ANG_SPEED    # rad/s yaw
        vy  = torch.zeros_like(lin)           # no lateral strafe
        return torch.stack([lin, vy, ang], dim=-1)

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
            "heading":  policy["heading"],
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
        base_action = self._base_cmd(action)
        obs, reward, terminated, truncated, info = self._env.step(base_action)
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
