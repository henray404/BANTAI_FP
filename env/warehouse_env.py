# warehouse_env.py
# Person 1 — Environment & Integration
#
# IMPORT RULE: No AppLauncher here. Imported by tests/test_env.py and scripts/run_env.py
# which own AppLauncher. See bugs_errors/2026-05-15_double-applaunch-crash.md.
#
# Interface contract v2 — pickup (DO NOT CHANGE without team discussion):
#   obs = {
#       "pixels":   Tensor(batch, 3, 64, 64),   # camera image, float in [0,1]
#       "position": Tensor(batch, 3),            # robot base xyz, env-local
#       "heading":  Tensor(batch, 2),            # [cos(yaw), sin(yaw)]
#       "goal":     Tensor(batch, 3),            # delivery zone xyz (curriculum: anneal→zeros)
#       "goal_id":  Tensor(batch, 3),            # one-hot category (replaced goal_emb 2026-06-08)
#       "ee_pos":   Tensor(batch, 3),            # end-effector xyz, base frame
#       "gripper":  Tensor(batch, 1),            # finger opening 0..1
#       "holding":  Tensor(batch, 1),            # 1.0 if target box grasped
#       "box_pos":  Tensor(batch, 3),            # target box xyz, env-local (UNANNEALED)
#   }
#   action_space = Box(-1, 1, shape=(6,))       # [base_lin, base_ang, ee_dx, ee_dy, ee_dz, gripper]
#
# NOTE: base velocity obs deliberately EXCLUDED (RSSM reconstructs base motion from pixels).
# Arm proprioception (ee_pos/gripper/holding) IS included — not inferable from base-mounted pixels.

"""Warehouse environment: ManagerBasedRLEnvCfg + Gymnasium wrapper for DreamerV3."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

import isaaclab.envs.mdp as mdp
from isaaclab.controllers import DifferentialIKControllerCfg
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
    out_of_bounds,
    time_penalty,
)
from env.reward_pickup import (
    approach_box_distance,
    carry_distance,
    grasp_success_reward,
    drop_penalty,
    pickup_delivered,
    pickup_delivered_reward,
)


# ── Constants ─────────────────────────────────────────────────────────
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
    """Return robot CHASSIS xyz relative to env origin, shape (num_envs, 3).

    Reads body_pos_w["base_link"], NOT root_pos_w. The Ridgeback-Franka is a fixed-root
    articulation: its root link is welded to `world`, so root_pos_w stays at the spawn pose
    while the chassis moves via the dummy base joints. Reading the root froze this obs at
    spawn (policy saw a robot that never moves). See IsaacLab issue #1268.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    idx = robot.body_names.index("base_link")  # moving chassis body (NOT the fixed root)
    return robot.data.body_pos_w[:, idx] - env.scene.env_origins


def goal_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return current per-env goal xyz, shape (num_envs, 3)."""
    if hasattr(env, "goal_pos"):
        return env.goal_pos
    return torch.zeros(env.num_envs, 3, device=env.device)


def goal_id(env: ManagerBasedRLEnv) -> torch.Tensor:
    """One-hot (num_envs, 3) commanded category [fragile, regular, heavy]. Reads env.goal_id_buf."""
    if hasattr(env, "goal_id_buf"):
        return env.goal_id_buf
    return torch.zeros(env.num_envs, 3, device=env.device)


def ee_position(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """End-effector (panda_hand) xyz in the base frame, shape (num_envs, 3)."""
    robot: Articulation = env.scene[asset_cfg.name]
    ee = robot.body_names.index("panda_hand")
    base = robot.body_names.index("base_link")
    return robot.data.body_pos_w[:, ee] - robot.data.body_pos_w[:, base]


def gripper_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Normalized finger opening (num_envs, 1) in [0,1] (0=closed, 1=open at 0.035 m)."""
    robot: Articulation = env.scene[asset_cfg.name]
    j = robot.joint_names.index("panda_finger_joint1")
    return (robot.data.joint_pos[:, j:j + 1] / 0.035).clamp(0.0, 1.0)


def holding_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(num_envs, 1) float: 1.0 if the target box is currently grasped. Reads env.holding."""
    if hasattr(env, "holding"):
        return env.holding.float().unsqueeze(-1)
    return torch.zeros(env.num_envs, 1, device=env.device)


def box_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Target box xyz, env-local (num_envs, 3). Reads env.box_pos (commanded by goal_id)."""
    if hasattr(env, "box_pos"):
        return env.box_pos
    return torch.zeros(env.num_envs, 3, device=env.device)


def robot_heading(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return [cos(yaw), sin(yaw)] heading, shape (num_envs, 2).

    Unit-circle encoding avoids the ±π discontinuity of raw yaw.
    Critical because robot spawns at random yaw — policy needs heading to orient itself.

    Reads body_quat_w["base_link"], NOT root_quat_w: the fixed articulation root never rotates
    while the chassis yaws via dummy_base_revolute_z, so root_quat_w froze this at spawn yaw.
    See IsaacLab issue #1268.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    idx = robot.body_names.index("base_link")  # moving chassis body (NOT the fixed root)
    quat = robot.data.body_quat_w[:, idx]  # (w, x, y, z)
    yaw = torch.atan2(
        2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
        1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2),
    )
    return torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)


# ── Actions ───────────────────────────────────────────────────────────
@configclass
class ActionsCfg:
    """Base velocity + arm IK + gripper. Internal action dim = 7 in declaration order:
    base_vel(3) + arm_ik(3) + gripper(1).

    The external policy action is (6,) [base_lin, base_ang, ee_dx, ee_dy, ee_dz, gripper];
    WarehouseGymEnv.step splits it and expands base to [vx, vy, wz] via _base_cmd, then
    concatenates [base3, ee3, grip1] into this 7-dim internal action. preserve_order keeps the
    base joint columns aligned with _base_cmd's output. The arm is driven by DifferentialIK
    (relative EE position, top-down orientation); gripper is binary open/close.
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

    # Arm: differential IK on the 7 panda joints. Relative-mode position command (3 dims:
    # dx,dy,dz in base frame); orientation held fixed (top-down) by the controller. Body
    # "panda_hand" is the Franka EE link. Mirrors the Isaac-Lift-Cube-Franka-v0 IK term.
    arm_ik = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(
            command_type="position", use_relative_mode=True, ik_method="dls"
        ),
        scale=1.0,
    )

    # Gripper: open/close as a binary action mapped to the two finger joints.
    gripper = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger_joint.*"],
        open_command_expr={"panda_finger_joint.*": 0.035},
        close_command_expr={"panda_finger_joint.*": 0.0},
    )


# ── Observations ──────────────────────────────────────────────────────
@configclass
class ObservationsCfg:
    """Returns dict obs matching interface contract (no concatenation)."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy obs group (v2): nav + manipulation keys, kept separate (no concatenation)."""

        # --- navigation ---
        pixels   = ObsTerm(func=camera_rgb)
        position = ObsTerm(func=robot_position)
        heading  = ObsTerm(func=robot_heading)    # [cos(yaw), sin(yaw)] — random-spawn orientation
        goal     = ObsTerm(func=goal_position)
        goal_id  = ObsTerm(func=goal_id)          # one-hot category (replaced goal_emb 2026-06-08)
        # --- manipulation ---
        ee_pos   = ObsTerm(func=ee_position)
        gripper  = ObsTerm(func=gripper_state)
        holding  = ObsTerm(func=holding_state)
        box_pos  = ObsTerm(func=box_position)

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
    """Staged pick-place reward (see spec §4). Phase switches on env.holding.

    Phase A (NOT holding): approach dense -0.01*dist(ee,box); grasp +5 one-shot.
    Phase B (holding):     carry dense -0.01*dist(box,zone);  deliver +10 while in zone.
    Always-on:             time -0.005; collision -5; drop -2 one-shot.
    """

    approach  = RewTerm(func=approach_box_distance,   weight=-0.01)
    grasp     = RewTerm(func=grasp_success_reward,    weight=5.0)
    carry     = RewTerm(func=carry_distance,          weight=-0.01)
    deliver   = RewTerm(func=pickup_delivered_reward, weight=10.0)
    time_pen  = RewTerm(func=time_penalty,            weight=-0.005)
    collision = RewTerm(func=collision_penalty,       weight=5.0)   # func returns 0/-1; weight=5 → -5
    drop      = RewTerm(func=drop_penalty,            weight=-2.0)


# ── Terminations ──────────────────────────────────────────────────────
@configclass
class TerminationsCfg:
    """Episode end conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success  = DoneTerm(func=pickup_delivered)   # held box inside its commanded color zone
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
        self.episode_length_s = 100.0  # 100s x 10Hz = 1000 steps (nav + grasp + carry + place)
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
        # Candidate goals: env-local xyz of each zone (from ZONE_SPECS).
        # ZONE_SPECS order = zone_A/fragile, zone_B/regular, zone_C/heavy → row c = category c.
        self._zone_pos = torch.tensor(
            [pos for _, _, pos in ZONE_SPECS], device=self.device, dtype=torch.float32
        )
        # Pickup runtime buffers (read by obs/reward fns + _carry_held_boxes).
        from env.warehouse_scene import TARGET_BOX_SPECS
        self._cat_names = ("fragile", "regular", "heavy")
        self._boxes_by_cat = {
            c: [n for (n, *_) in TARGET_BOX_SPECS if n.startswith(c)] for c in self._cat_names
        }
        # env-local resting z per box (shelf surface + size/2) — lift is measured against this,
        # NOT absolute world z (a shelf box already sits ~0.8m up).
        self._box_rest_z = {n: pos[2] for (n, _s, _m, pos) in TARGET_BOX_SPECS}
        # Box edge length per target — grasp uses distance to box SURFACE (size-aware), not center.
        self._box_size = {n: s for (n, s, _m, _pos) in TARGET_BOX_SPECS}
        N, dev = self.num_envs, self.device
        self.goal_id_buf = torch.zeros(N, 3, device=dev)
        self.box_cat_idx = torch.zeros(N, dtype=torch.long, device=dev)
        self.target_box_name = ["" for _ in range(N)]
        self.box_pos = torch.zeros(N, 3, device=dev)
        self.ee_pos = torch.zeros(N, 3, device=dev)
        self.holding = torch.zeros(N, dtype=torch.bool, device=dev)
        self.grasp_event = torch.zeros(N, dtype=torch.bool, device=dev)
        self.drop_event = torch.zeros(N, dtype=torch.bool, device=dev)
        self._sample_targets(torch.arange(N, device=dev))

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
        # Ridgeback-Franka: 3 base + 7 arm + 2 finger = 12 joints. The (6,) action drives base(3)
        # via _base_cmd, arm(7) via DifferentialIK, and the 2 fingers via the binary gripper term.
        robot: Articulation = self.scene["robot"]
        n_joints = robot.num_joints
        if n_joints < 12:
            raise RuntimeError(
                f"Expected >=12 joints (3 base + 7 arm + 2 finger) on the Ridgeback-Franka, "
                f"found {n_joints}. Check ActionsCfg joint_names match the dummy_base_*/panda_* "
                "joints, or verify ridgeback_franka.usd loaded (Nucleus unreachable -> 0 joints)."
            )

    def _sample_targets(self, env_ids: torch.Tensor) -> None:
        """Per env: pick a commanded category → goal_id one-hot, a target box, and matching zone."""
        from env.curriculum import goal_id_onehot
        if env_ids.numel() == 0:
            return
        for e in env_ids.tolist():
            c = int(torch.randint(0, 3, (1,), device=self.device))
            self.box_cat_idx[e] = c
            names = self._boxes_by_cat[self._cat_names[c]]
            self.target_box_name[e] = names[int(torch.randint(0, len(names), (1,)))]
            self.goal_pos[e] = self._zone_pos[c]   # zone order == category order
        self.goal_id_buf[env_ids] = goal_id_onehot(self.box_cat_idx[env_ids], num_cats=3)
        self.holding[env_ids] = False

    def _randomize_box_poses(self, env_ids: torch.Tensor) -> None:
        """Random x,y jitter for the 18 bottom-shelf boxes within their shelf-deck area.

        Called AFTER super()._reset_idx() so it overrides scene.reset() default positions.
        Each box keeps its shelf resting z (shelf surface + size/2 + 5cm drop) but gets a new
        random x,y offset. Velocities zeroed to prevent carry-over from previous episode.
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
        """Sample targets+goals, reset scene, randomize box poses, then refresh target box pos."""
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._sample_targets(env_ids_t)
        super()._reset_idx(env_ids)
        self._randomize_box_poses(env_ids_t)  # after super() so scene.reset() doesn't undo it
        self._refresh_target_box_pos(env_ids_t)

    def _refresh_target_box_pos(self, env_ids: torch.Tensor | None = None) -> None:
        """Write each env's commanded box xyz (env-local) into self.box_pos."""
        ids = range(self.num_envs) if env_ids is None else env_ids.tolist()
        for e in ids:
            box = self.scene[self.target_box_name[e]]
            self.box_pos[e] = box.data.root_pos_w[e] - self.scene.env_origins[e]

    def update_grasp(self) -> None:
        """Evaluate grasp-success / drop, set holding + one-shot events, kinematically carry held box.

        Called by WarehouseGymEnv.step after the sim step. Sets self.grasp_event / self.drop_event
        (one-step flags read by reward terms) and self.holding (read by obs + reward gating).
        """
        from env.grasp import grasp_success, grasp_lost
        self._refresh_target_box_pos()
        robot: Articulation = self.scene["robot"]
        ee = robot.body_names.index("panda_hand")
        base = robot.body_names.index("base_link")
        self.ee_pos = robot.data.body_pos_w[:, ee] - robot.data.body_pos_w[:, base]
        ee_world = robot.data.body_pos_w[:, ee]
        j = robot.joint_names.index("panda_finger_joint1")
        closed = robot.data.joint_pos[:, j] < 0.0175   # < half of open (0.035)
        # Grasp on proximity to box SURFACE + gripper closed (boxes are larger than the gripper —
        # no physical enclosure/lift required; carry is kinematic). box_half = box edge / 2.
        box_half = torch.tensor(
            [self._box_size[self.target_box_name[e]] * 0.5 for e in range(self.num_envs)],
            device=self.device,
        )
        newly = grasp_success(self.ee_pos, self.box_pos, closed, box_half) & (~self.holding)
        lost = grasp_lost(self.holding, self.ee_pos, self.box_pos, box_half) | (~closed & self.holding)
        self.grasp_event = newly
        self.drop_event = lost & (~self._box_in_any_zone())
        self.holding = (self.holding | newly) & (~lost)
        self._carry_held_boxes(ee_world)

    def _box_in_any_zone(self) -> torch.Tensor:
        """(N,) bool: target box xy within 1.5 m of its commanded zone center (env-local)."""
        return torch.norm(self.box_pos[:, :2] - self.goal_pos[:, :2], dim=-1) < 1.5

    def _carry_held_boxes(self, ee_world: torch.Tensor) -> None:
        """Teleport each held box to the EE (kinematic carry) so physics grip isn't required."""
        for e in range(self.num_envs):
            if not bool(self.holding[e]):
                continue
            box = self.scene[self.target_box_name[e]]
            state = box.data.root_state_w[e:e + 1].clone()
            state[:, 0:3] = ee_world[e:e + 1]
            state[:, 7:13] = 0.0  # zero linear + angular velocity
            box.write_root_state_to_sim(state, env_ids=torch.tensor([e], device=self.device))


# ── Gymnasium Wrapper ─────────────────────────────────────────────────
class WarehouseGymEnv(gym.Env):
    """Gymnasium-style wrapper around `WarehouseRLEnv`.

    Exposes:
        action_space      = Box(-1, 1, shape=(6,))  [base_lin, base_ang, ee_dx, ee_dy, ee_dz, gripper]
        observation_space = Dict(pixels, position, heading, goal, goal_id,
                                 ee_pos, gripper, holding, box_pos)

    step() splits the (6,) action: base(2) → _base_cmd → [vx,vy,wz]; ee(3) IK delta; gripper(1),
    concatenated to the (7,) internal joint action, then calls update_grasp(). Returns batched
    tensors (num_envs, ...); single-env consumers should set num_envs=1.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg: WarehouseEnvCfg | None = None, render_mode: str | None = None):
        """Build underlying RL env and Gym-style spaces."""
        self.cfg = cfg if cfg is not None else WarehouseEnvCfg()
        self._env = WarehouseRLEnv(cfg=self.cfg, render_mode=render_mode)
        self.num_envs: int = self._env.num_envs
        self.device = self._env.device
        # Moving-chassis body index, cached once. Fixed-root robot: root_pos_w never moves while
        # base_link does (IsaacLab #1268); _base_cmd reads base_link yaw for body-frame drive.
        self._base_link_idx: int = self._env.scene["robot"].body_names.index("base_link")
        # Chassis yaw RELATIVE to the root/prismatic frame. The dummy prismatic joints translate in
        # the articulation-root frame (root pose is yaw-randomized on reset), so drive must project
        # by the revolute_z joint angle, NOT the absolute base_link world yaw — otherwise root_yaw
        # is double-counted. See bugs_errors/2026-06-16_base-drive-doublecounts-spawn-yaw.md.
        self._revolute_z_idx: int = self._env.scene["robot"].joint_names.index(
            "dummy_base_revolute_z_joint"
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "pixels":   spaces.Box(0.0, 1.0, shape=(3, IMG_HW, IMG_HW), dtype=np.float32),
                "position": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "heading":  spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),   # [cos(yaw), sin(yaw)]
                "goal":     spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "goal_id":  spaces.Box(0.0, 1.0, shape=(3,), dtype=np.float32),    # one-hot category
                "ee_pos":   spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "gripper":  spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "holding":  spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "box_pos":  spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            }
        )

    def _base_yaw(self) -> torch.Tensor:
        """Chassis yaw (rad) RELATIVE to the root/prismatic frame = revolute_z joint angle, (num_envs,).

        The dummy prismatic joints translate in the articulation-root frame (root pose is
        yaw-randomized on reset), so projecting the drive command by this joint angle yields
        world motion = root_yaw + revolute_z = base_link world yaw (the facing direction). Using
        the absolute base_link world yaw would double-count root_yaw — see
        bugs_errors/2026-06-16_base-drive-doublecounts-spawn-yaw.md.
        """
        return self._env.scene["robot"].data.joint_pos[:, self._revolute_z_idx]

    def _base_cmd(self, action: torch.Tensor) -> torch.Tensor:
        """Map [linear, angular] in [-1,1] → holonomic base joint velocities [vx, vy, wz].

        Order matches ActionsCfg.base_vel joint_names (preserve_order=True):
        [dummy_base_prismatic_x, dummy_base_prismatic_y, dummy_base_revolute_z].

        The dummy prismatic joints translate in the WORLD frame (they precede revolute_z in the
        chain world→prismatic_x→prismatic_y→revolute_z→base_link), so a raw [lin, 0, ang] always
        slides the robot along world +x regardless of heading. We project the linear command by
        the current chassis yaw so "forward" follows the robot's facing direction (body-frame
        drive on a world-frame base), keeping the (2,) contract (no strafe term). See IsaacLab
        discussion #2664 — maintainer: transform control vectors by orientation before applying
        them to the X/Y dummy joints.
        """
        lin = action[:, 0] * MAX_LIN_SPEED    # commanded forward speed (m/s, body frame)
        ang = action[:, 1] * MAX_ANG_SPEED    # rad/s yaw
        yaw = self._base_yaw()                # current chassis heading
        vx  = lin * torch.cos(yaw)            # → world-x velocity target (prismatic_x)
        vy  = lin * torch.sin(yaw)            # → world-y velocity target (prismatic_y)
        return torch.stack([vx, vy, ang], dim=-1)

    _OBS_KEYS = ("pixels", "position", "heading", "goal", "goal_id",
                 "ee_pos", "gripper", "holding", "box_pos")

    def _unwrap_obs(self, obs: dict) -> dict[str, torch.Tensor]:
        """Pull terms out of obs['policy'] dict to match the v2 interface contract."""
        policy = obs["policy"]
        if not isinstance(policy, dict):
            raise RuntimeError(
                "ObservationsCfg returned non-dict 'policy'. "
                "Ensure PolicyCfg.concatenate_terms = False."
            )
        return {k: policy[k] for k in self._OBS_KEYS}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reset all sub-envs; return (obs_dict, info)."""
        if seed is not None:
            torch.manual_seed(seed)
        obs, info = self._env.reset()
        return self._unwrap_obs(obs), info

    def step(self, action):
        """Apply (6,) action [base_lin, base_ang, ee_dx, ee_dy, ee_dz, gripper]."""
        from env.action_pickup import split_action
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(self.device, dtype=torch.float32)
        if action.ndim == 1:
            action = action.unsqueeze(0).expand(self.num_envs, -1)
        action = action.clamp(-1.0, 1.0).to(self.device, dtype=torch.float32)
        base2, ee3, grip1 = split_action(action)
        base3 = self._base_cmd(base2)                       # (N,3) base joint velocities
        internal = torch.cat([base3, ee3, grip1], dim=-1)   # (N,7) base(3)+ik(3)+gripper(1)
        obs, reward, terminated, truncated, info = self._env.step(internal)
        self._env.update_grasp()                            # holding + grasp/drop events, carry box
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
