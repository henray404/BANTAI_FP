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

import math
import pathlib

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

import isaaclab.envs.mdp as mdp
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
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
    collided,
    collision_penalty,
    idle_penalty,
    no_grasp_timeout,
    out_of_bounds,
    time_penalty,
    under_rack_penalty,
)
from env.reward_pickup import (
    approach_box_distance,
    box_dropped,
    carry_distance,
    carry_regress_penalty,
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

# Carry model (team decision 2026-06-21, Henry): default "physics" — on grab the box is welded to
# the chassis with a UsdPhysics.FixedJoint (env.attach) and STAYS VISIBLE, physically carried (it
# rides in front of the robot and moves with it). "kinematic" (hide + teleport each step) is kept
# as a lighter fallback. See docs/progress_p4.md.
CARRY_MODE = "physics"

# Carry anchor: a held box rides at this offset from the chassis (base_link) — in front + raised —
# so it is VISIBLY lifted in front of the robot instead of snapping into the tucked hand/body mesh.
GRIP_FWD = 0.6   # metres in front of the chassis (body +x)
GRIP_UP  = 0.7   # metres above the chassis origin

# ── Active-arm control (Lane B) ───────────────────────────────────────
# Port of the VERIFIED scripts/drive_env_v2.py teleop control into training: an absolute-hold EE
# target accumulator + DLS pose IK + the v2 clamp stack, replacing the stock relative-mode IK term
# (which ratchet/drifts — see bugs_errors/2026-06-22_ik-target-root-frame-drag-flail.md, Lane B).
# Engaged for curriculum stage >= 2 (grasp needs an active arm); stage 1 keeps the frozen arm.
# Rate note: v2 ran these per 200 Hz physics step; here _drive_arm runs once per 10 Hz CONTROL step
# and the joint PD tracks the target across the 20 substeps — so smoothing/step caps are retuned for
# 10 Hz (looser than v2's 200 Hz values), not copied verbatim. P1 tunes in sim via drive_env_v2.py.
ARM_IK_DAMP        = 0.1    # DLS lambda for top-down pose IK (no wrist-flip collapse)
ARM_SMOOTH_BETA    = 0.5    # EXP low-pass on IK joint targets per control step (1.0 = raw)
ARM_MAX_JOINT_STEP = 0.25   # rad cap on |target - current| per control step (flip-safety, ~2.5 rad/s)
ARM_JOINT_MARGIN   = 0.95   # clamp joints to this fraction of the hard USD range, centred
ARM_REACH_X_BACK   = 0.0    # min EE-target x (base frame): no backward reach into base/rack
ARM_REACH_Z_MIN    = 0.05   # min EE-target z (base frame): floor
ARM_REACH_Z_MAX    = 1.1    # max EE-target z (base frame): no over-reach up
ARM_LEAD_M         = 0.08   # max EE-target lead ahead of the live EE (anti-overrun leash)
ARM_REACH_R        = 0.95   # radial clamp r_max (|EE - shoulder|); 0 = off
ARM_REACH_RMIN     = 0.50   # radial clamp r_min
# calib/arm_calib.yaml (written by scripts/calibrate_arm.py --auto) overrides the radial defaults,
# same file drive_env_v2.py reads — keeps training on the robot's fitted reach envelope.
_ARM_CALIB_PATH = pathlib.Path(__file__).resolve().parents[1] / "calib" / "arm_calib.yaml"
if _ARM_CALIB_PATH.exists():
    try:
        import yaml as _yaml
        _ac = (_yaml.safe_load(_ARM_CALIB_PATH.read_text(encoding="utf-8")) or {}).get("arm_calib", {})
        ARM_REACH_R = float(_ac.get("reach_r", ARM_REACH_R))
        ARM_REACH_RMIN = float(_ac.get("reach_rmin", ARM_REACH_RMIN))
    except Exception:  # malformed/locked calib must never block env construction
        pass

# Idle/stuck reset: end the episode if the base hasn't translated more than STUCK_MOVE_EPS_M per
# control step for STUCK_STEPS consecutive steps (~45 s at 10 Hz control) — frees a robot wedged
# against a wall/rack instead of wasting the full 100 s episode standing still.
STUCK_MOVE_EPS_M = 0.02
STUCK_STEPS = 450

# Idle penalty (freeze-trap breaker): once the base has been idle for IDLE_PENALTY_STEPS consecutive
# control steps (~5 s at 10 Hz), apply a per-step cost so STANDING STILL is strictly more expensive
# than careful movement. Fires long before STUCK_STEPS (the hard reset). See warehouse_reward.idle_penalty.
IDLE_PENALTY_STEPS = 50

# Heavy "too slow" penalty: a second, steeper idle tier that fires after IDLE_SLOW_STEPS consecutive
# idle control steps (~30 s at 10 Hz) — escalates above the 5 s soft idle nudge but still before the
# STUCK_STEPS (~45 s) hard reset, so a robot loitering for half a minute pays a real cost first.
IDLE_SLOW_STEPS = 300

# Carry-regress penalty: while HOLDING, count consecutive control steps where the box gets no closer
# to its goal zone (backing up / dawdling). Past CARRY_REGRESS_STEPS (~5s @ 10Hz) a per-step cost
# fires so the robot doesn't reverse forever on the way to the finish zone. Progress = the box→goal
# distance shrinks by > CARRY_PROGRESS_EPS_M in a step.
CARRY_REGRESS_STEPS = 50
CARRY_PROGRESS_EPS_M = 0.02

# Reset-to-checkpoint (num_envs==1): while carrying, snapshot the sim state at grasp and each time
# the box crosses a closer CHECKPOINT_RING_M ring to the goal (2/4/6 m…). On a FAILURE reset
# (crash/drop/stuck/bounds) the episode ends cleanly (done=True) and the NEXT episode RESTARTS from
# the last checkpoint instead of the far spawn — speeds up training without breaking the world model
# (a clean done, not a mid-episode teleport). Success / time-out / no-checkpoint → fresh spawn.
CHECKPOINT_RING_M = 2.0


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
    """Return current per-env goal xyz (num_envs, 3), scaled by the curriculum goal-leak alpha.

    goal_alpha defaults to 1.0 (full leak = legacy behavior); STAGE_ANNEAL drives it toward 0 so
    the policy must deliver from goal_id + pixels alone. box_pos obs stays UNANNEALED (contract).
    """
    if hasattr(env, "goal_pos"):
        from env.curriculum import anneal_goal
        return anneal_goal(env.goal_pos, getattr(env, "goal_alpha", 1.0))
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


def stuck_timeout(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(num_envs,) bool: base idle (no translation) for >= STUCK_STEPS steps. Reads env._stuck_steps."""
    if hasattr(env, "_stuck_steps"):
        return env._stuck_steps >= STUCK_STEPS
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def failure_reset(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(N,) bool: ANY failure-reset condition this step — crash, dropped carried box, no-grasp
    timeout, out-of-bounds, or stuck. Excludes success (delivery) and the time-limit truncation.
    The failure DoneTerms fire on subsets of this; failure_penalty applies the high cost on it.
    """
    fail = collided(env) | no_grasp_timeout(env) | out_of_bounds(env) | stuck_timeout(env)
    if hasattr(env, "drop_event"):
        fail = fail | env.drop_event
    return fail


def failure_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """+1 on any failure-reset step (use with a large NEGATIVE weight = the 'high punishment')."""
    return failure_reset(env).float()


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
    """Base velocity + gripper. Internal action dim = 4 in declaration order: base_vel(3) + gripper(1).

    The external policy action is (6,) [base_lin, base_ang, ee_dx, ee_dy, ee_dz, gripper];
    WarehouseGymEnv.step splits it, expands base to [vx, vy, wz] via _base_cmd, and concatenates
    [base3, grip1] into this 4-dim internal action. preserve_order keeps the base joint columns
    aligned with _base_cmd's output.

    The ARM is NOT a manager action term: the stock DifferentialInverseKinematicsActionCfg is
    relative-mode and ratchet/drifts (Lane B bug doc 2026-06-22). It is driven manually by
    WarehouseRLEnv._drive_arm (absolute-hold EE target + clamp stack, ported from the verified
    drive_env_v2.py) for stage >= 2, and frozen (pinned to home) for stage 1. gripper is binary.
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


# ── Config loading (tunable YAML in configs/) ─────────────────────────
_CONFIG_DIR = pathlib.Path(__file__).resolve().parents[1] / "configs"
_REWARD_CFG_PATH = _CONFIG_DIR / "reward_weights.yaml"
_ENV_CFG_PATH = _CONFIG_DIR / "env_config.yaml"


def _read_yaml(path: pathlib.Path) -> dict:
    """Parse a YAML file to a dict (pyyaml, ruamel fallback); {} if the file is absent."""
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
        return yaml.safe_load(text) or {}
    except ImportError:
        import ruamel.yaml as ryaml
        return ryaml.YAML(typ="safe").load(text) or {}


def _load_reward_weights() -> dict:
    """Return the `rewards:` mapping from configs/reward_weights.yaml ({} if file/section absent).

    Tunable weights live in the YAML so shaping can be tweaked without editing code. Missing file
    or keys fall back to the RewardsCfg Python defaults (applied in WarehouseEnvCfg.__post_init__).
    """
    return _read_yaml(_REWARD_CFG_PATH).get("rewards", {}) or {}


def _load_env_config() -> dict:
    """Return the live-tunable env knobs from configs/env_config.yaml, flattened ({} if absent).

    Only a curated subset of that (otherwise documentation) file is honored — episode length,
    decimation, sim dt, and base speed. Keys absent from the YAML are dropped so callers keep their
    Python defaults. Applied in WarehouseEnvCfg.__post_init__.
    """
    data = _read_yaml(_ENV_CFG_PATH)
    env = data.get("env", {}) or {}
    sim = data.get("simulation", {}) or {}
    rob = data.get("robot", {}) or {}
    out = {
        "decimation": env.get("decimation"),
        "episode_length_s": env.get("episode_length_s"),
        "sim_dt": sim.get("dt"),
        "max_lin_speed": rob.get("max_lin_speed"),
        "max_ang_speed": rob.get("max_ang_speed"),
    }
    return {k: v for k, v in out.items() if v is not None}


@configclass
class RewardsCfg:
    """Staged pick-place reward (see spec §4). Phase switches on env.holding.

    Phase A (NOT holding): approach dense -0.02*dist(ee,box); grasp +5 one-shot.
    Phase B (holding):     carry dense -0.02*dist(box,zone);  deliver +10 while in zone.
    Always-on:             time -0.005; collision -2; idle -0.02; drop -2 one-shot.

    Anti-freeze tuning (2026-06-21): approach/carry pull DOUBLED (-0.01→-0.02) so the dense draw to
    the goal out-weighs the collision fear; collision HALVED (-5→-2) so the robot dares to explore
    around the racks instead of freezing; idle -0.02/step makes standing still strictly worse than
    careful movement. See docs/progress_p4.md.
    """

    approach  = RewTerm(func=approach_box_distance,   weight=-0.02)
    grasp     = RewTerm(func=grasp_success_reward,    weight=5.0)
    carry     = RewTerm(func=carry_distance,          weight=-0.02)
    deliver   = RewTerm(func=pickup_delivered_reward, weight=10.0)
    time_pen  = RewTerm(func=time_penalty,            weight=-0.005)
    collision = RewTerm(func=collision_penalty,       weight=2.0)   # func returns 0/-1; weight=2 → -2
    idle      = RewTerm(func=idle_penalty, params={"idle_steps": IDLE_PENALTY_STEPS}, weight=0.02)  # 0/-1 → -0.02 @5s
    idle_slow = RewTerm(func=idle_penalty, params={"idle_steps": IDLE_SLOW_STEPS},    weight=0.1)   # 0/-1 → -0.1 @30s (heavy "too slow")
    drop      = RewTerm(func=drop_penalty,            weight=-2.0)
    # One-shot HIGH cost on ANY failure reset (crash / dropped box / no-grasp timeout / bounds /
    # stuck) — the "punishment yang tinggi tiap reset". Fixed (not escalating: a non-stationary
    # reward would hurt the DreamerV3 world model). Tunable in configs/reward_weights.yaml.
    failure    = RewTerm(func=failure_penalty,        weight=-15.0)
    under_rack = RewTerm(func=under_rack_penalty,     weight=2.0)    # 0/-1 → -2/step while under a rack
    # While carrying: -1/step once the box has gone CARRY_REGRESS_STEPS steps without nearing the
    # zone (backing up / dawdling). x(-1)*weight → -0.1/step. Stops "kelamaan mundur" to the finish.
    carry_regress = RewTerm(func=carry_regress_penalty,
                            params={"regress_steps": CARRY_REGRESS_STEPS}, weight=0.1)


# ── Terminations ──────────────────────────────────────────────────────
@configclass
class TerminationsCfg:
    """Episode end conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success  = DoneTerm(func=pickup_delivered)   # held box inside its commanded color zone
    bounds   = DoneTerm(func=out_of_bounds, params={"half_extent_x": 9.5, "half_extent_y": 14.5})
    stuck    = DoneTerm(func=stuck_timeout)      # base idle ~45 s → reset (wedged in a wall/rack)
    crashed  = DoneTerm(func=collided)           # chassis contact > 50 N → reset (nabrak)
    dropped  = DoneTerm(func=box_dropped)        # carried box fell to the floor → reset
    no_grasp = DoneTerm(func=no_grasp_timeout)   # 30 s with no grasp → reset (arm-first focus)


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
        # Live env knobs from configs/env_config.yaml override the defaults just set (missing keys
        # keep them). Base speed lives in module globals (read at call time in _base_cmd), so update
        # them here. ponytail: globals are process-wide — fine at num_envs=1, last cfg wins if many.
        _e = _load_env_config()
        self.decimation = int(_e.get("decimation", self.decimation))
        self.episode_length_s = float(_e.get("episode_length_s", self.episode_length_s))
        self.sim.dt = float(_e.get("sim_dt", self.sim.dt))
        if "max_lin_speed" in _e:
            globals()["MAX_LIN_SPEED"] = float(_e["max_lin_speed"])
        if "max_ang_speed" in _e:
            globals()["MAX_ANG_SPEED"] = float(_e["max_ang_speed"])
        self.sim.render_interval = self.decimation
        # Arm + contact stability (Ridgeback-Franka): reduce noisy base/arm velocities.
        self.sim.physx.enable_external_forces_every_iteration = True
        self.viewer.eye = (0.0, -20.0, 18.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)
        # Reward/penalty weights are tunable via configs/reward_weights.yaml — edit + restart to
        # retune shaping without touching code. Missing file/keys keep the RewardsCfg defaults.
        _w = _load_reward_weights()
        for _name in ("approach", "grasp", "carry", "deliver", "time_pen",
                      "collision", "idle", "idle_slow", "drop", "failure", "under_rack",
                      "carry_regress"):
            if _name in _w:
                getattr(self.rewards, _name).weight = float(_w[_name])
        if "idle_steps" in _w:
            self.rewards.idle.params["idle_steps"] = int(_w["idle_steps"])
        if "idle_slow_steps" in _w:
            self.rewards.idle_slow.params["idle_steps"] = int(_w["idle_slow_steps"])


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
        self.ee_pos = torch.zeros(N, 3, device=dev)        # base-frame delta (proprioception OBS)
        self.ee_pos_world = torch.zeros(N, 3, device=dev)  # env-local world (REWARD shaping; frame-matches box_pos)
        self.holding = torch.zeros(N, dtype=torch.bool, device=dev)
        self.grasp_event = torch.zeros(N, dtype=torch.bool, device=dev)
        self.drop_event = torch.zeros(N, dtype=torch.bool, device=dev)
        self.deliver_event = torch.zeros(N, dtype=torch.bool, device=dev)  # one-shot: first delivered step
        self._delivered = torch.zeros(N, dtype=torch.bool, device=dev)     # latch: delivered this episode
        self._ever_grasped = torch.zeros(N, dtype=torch.bool, device=dev)  # latch: grasped >=1x this episode (no-grasp timeout)
        # Idle/stuck detector (see stuck_timeout termination): consecutive idle steps + prev base xy.
        self._stuck_steps = torch.zeros(N, device=dev)
        self._prev_base_xy = torch.zeros(N, 2, device=dev)
        # Carry-regress detector: consecutive holding steps with no progress toward the goal zone.
        self._carry_regress_steps = torch.zeros(N, device=dev)
        self._prev_carry_dist = torch.zeros(N, device=dev)
        # Reset-to-checkpoint (num_envs==1): a saved sim_state blob + the closest 2m ring captured.
        self._ckpt_valid = torch.zeros(N, dtype=torch.bool, device=dev)
        self._ckpt_ring = torch.zeros(N, device=dev)
        self._checkpoint = None   # recording.sim_state blob for env 0
        # Curriculum 4-stage manager (env.curriculum). Default STAGE_FULL = legacy full chain.
        # P3/P5 drive transitions via set_stage()/set_goal_alpha(); P4 provides the mechanism only.
        from env.curriculum import STAGE_FULL
        self.stage: int = STAGE_FULL
        self._anneal_alpha: float = 1.0   # scheduled goal-leak for STAGE_ANNEAL (set externally)
        self.goal_alpha: float = 1.0      # effective goal-leak applied in goal_position()
        # Physics-grasp bookkeeping (CARRY_MODE == "physics"): cache USD stage + per-env hand prim.
        self._usd_stage = None
        self._hand_prim_path: dict[int, str | None] = {}
        self._sample_targets(torch.arange(N, device=dev))
        self._disable_arm_collisions()   # frozen arm is non-functional here — stop it snagging racks

    # ── Curriculum API (mechanism for P3/P5; they own the transition policy) ──────────
    def set_stage(self, stage: int) -> None:
        """Set curriculum stage 1..4. Stages 1-3 force full goal leak; 4 honors goal_alpha.

        Per-stage spawn / pre-grasp take effect on the NEXT _reset_idx; the goal-leak applies
        immediately via goal_position().
        """
        from env.curriculum import resolve_goal_alpha, validate_stage
        self.stage = validate_stage(stage)
        self.goal_alpha = resolve_goal_alpha(self.stage, self._anneal_alpha)

    def set_goal_alpha(self, alpha: float) -> None:
        """Schedule the STAGE_ANNEAL goal-leak in [0,1] (ignored on stages 1-3)."""
        from env.curriculum import resolve_goal_alpha
        self._anneal_alpha = float(alpha)
        self.goal_alpha = resolve_goal_alpha(self.stage, self._anneal_alpha)

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
        from env.curriculum import goal_id_onehot, stage_is_pregrasped
        if env_ids.numel() == 0:
            return
        for e in env_ids.tolist():
            c = int(torch.randint(0, 3, (1,), device=self.device))
            self.box_cat_idx[e] = c
            names = self._boxes_by_cat[self._cat_names[c]]
            self.target_box_name[e] = names[int(torch.randint(0, len(names), (1,)))]
            self.goal_pos[e] = self._zone_pos[c]   # zone order == category order
        self.goal_id_buf[env_ids] = goal_id_onehot(self.box_cat_idx[env_ids], num_cats=3)
        # Stage 1 (Nav-only) spawns the box already held; the physical pre-grasp (snap+weld) is
        # applied in _pregrasp_box after the scene reset. All other stages start not-holding.
        self.holding[env_ids] = stage_is_pregrasped(self.stage)
        self._delivered[env_ids] = False   # clear the per-episode delivery latch
        self._ever_grasped[env_ids] = self.holding[env_ids]  # pregrasped counts as grasped (no timeout)

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
        """Release held boxes, sample targets+goals, reset scene, randomize, apply stage reset."""
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        # Reset-to-checkpoint decision (num_envs==1): a FAILURE reset with a saved checkpoint resumes
        # from it; success / time-out / no-checkpoint fall through to a fresh spawn. Decide BEFORE the
        # fresh flow clobbers state; preserve the carried-box target identity to revert after.
        _do_restore = False
        _saved_target = None
        if (self.num_envs == 1 and 0 in env_ids_t.tolist() and bool(self._ckpt_valid[0])
                and bool(failure_reset(self)[0])):
            _do_restore = True
            _saved_target = (self.target_box_name[0], self.box_cat_idx[0].clone(),
                             self.goal_pos[0].clone(), self.goal_id_buf[0].clone())
        if CARRY_MODE == "physics":
            self._detach_boxes(env_ids_t)   # drop any box welded last episode before re-sampling
        else:
            self._set_box_visibility(env_ids_t, visible=True)   # un-hide any box hidden during carry
            self._set_box_collision(env_ids_t, enabled=True)    # re-enable collision after carry
        self._sample_targets(env_ids_t)
        super()._reset_idx(env_ids)
        self._randomize_box_poses(env_ids_t)  # after super() so scene.reset() doesn't undo it
        self._refresh_target_box_pos(env_ids_t)
        self._apply_stage_reset(env_ids_t)     # stage-2 spawn-near-box / stage-1 pre-grasp
        # Reset the idle/stuck detector for these envs (prev xy = the new spawn pose).
        self._stuck_steps[env_ids_t] = 0.0
        _robot = self.scene["robot"]
        _bidx = _robot.body_names.index("base_link")
        self._prev_base_xy[env_ids_t] = _robot.data.body_pos_w[env_ids_t, _bidx, :2]
        self._carry_regress_steps[env_ids_t] = 0.0
        self._prev_carry_dist[env_ids_t] = torch.norm(
            self.box_pos[env_ids_t, :2] - self.goal_pos[env_ids_t, :2], dim=-1
        )
        self._reanchor_arm(env_ids_t)   # active arm: re-home the EE target/smooth buffers at spawn
        # Fresh spawn just ran for ALL envs (a valid baseline). For a checkpoint restore, overwrite
        # env-0 with the saved state; if it throws, the env stays in the fresh spawn (fail-safe).
        if self.num_envs == 1 and 0 in env_ids_t.tolist():
            if _do_restore:
                try:
                    self._restore_checkpoint_env0(_saved_target)
                except Exception as _e:   # bad checkpoint -> drop it, keep the fresh spawn
                    self._ckpt_valid[0] = False
                    print(f"[checkpoint] restore failed ({_e}); fresh spawn", flush=True)
            else:
                self._ckpt_valid[0] = False   # fresh episode -> no checkpoint until it grasps again

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
        # ee in env-local world (same frame as box_pos) so the approach reward's distance actually
        # shrinks as the robot nears the box. self.ee_pos (base-frame delta) is ~constant with the
        # arm frozen, so reusing it for shaping gave a DEAD gradient — see C1 in
        # docs/project/training_readiness_2026-06-22.md.
        self.ee_pos_world = ee_world - self.scene.env_origins
        j = robot.joint_names.index("panda_finger_joint1")
        closed = robot.data.joint_pos[:, j] < 0.0175   # < half of open (0.035)
        # MAGNETIC pickup: the arm is frozen (WarehouseGymEnv.step zeros the EE action), so the
        # robot never reaches/knocks the box — it drives up, stops in front, and the box is grabbed
        # on PROXIMITY of the (static) hand to the box surface + gripper closed. box_half = box
        # edge / 2 makes the grip "span" the box size (no physical enclosure; boxes > gripper).
        # Compare hand and box in the SAME (world) frame; box_pos is env-local so add env_origins.
        # (Prior code compared base-frame ee_pos to env-local box_pos — mismatched frames, never fired.)
        box_world = self.box_pos + self.scene.env_origins
        box_half = torch.tensor(
            [self._box_size[self.target_box_name[e]] * 0.5 for e in range(self.num_envs)],
            device=self.device,
        )
        newly = grasp_success(ee_world, box_world, closed, box_half) & (~self.holding)
        # "Nyantol": the box is FixedJoint-welded to base_link AT THE HAND and the arm freezes at the
        # grab pose, so it rides with the robot and CANNOT fall via gripper-open (that release is
        # removed). The only loss is the safety: a held box that SEPARATED from the hand surface
        # (weld broke — physics anomaly) = a true drop → reset. grasp_lost is valid now because the
        # box is snapped to the hand (_grip_anchor_world = panda_hand), not a separate carry anchor.
        in_zone = self._box_in_any_zone()
        box_fell = grasp_lost(self.holding, ee_world, box_world, box_half)
        self.grasp_event = newly
        self.drop_event = box_fell & (~in_zone)
        self.holding = (self.holding | newly) & (~box_fell)
        self._ever_grasped = self._ever_grasped | newly   # latch for the no-grasp timeout
        # Active arm: snapshot the arm pose at the instant of grab so the carry can freeze the hand
        # exactly where the box was welded to base_link (else the hand drifts off the welded box).
        if bool(newly.any()) and getattr(self, "_arm_ready", False):
            self._arm_freeze_pose[newly] = robot.data.joint_pos[:, self._arm_joint_ids][newly]
        # Delivery: box (still held) sits in its commanded zone. One-shot the first such step so
        # consumers (record summary, P3) get a clean event; latch _delivered so it fires only once.
        delivered_now = self.holding & in_zone
        self.deliver_event = delivered_now & (~self._delivered)
        self._delivered = self._delivered | delivered_now
        anchor = self._grip_anchor_world()   # carry point in front of + above the chassis
        if CARRY_MODE == "physics":
            self._apply_physics_grasp(newly, box_fell, anchor)
        else:
            # Hidden-kinematic carry: on grab HIDE the box + DISABLE its collision (so the box that
            # follows the robot doesn't ram the racks) — no render, no physics push, cheaper; on
            # release show it + re-enable collision. Then teleport held boxes to follow the robot.
            grabbed = torch.nonzero(newly, as_tuple=False).flatten()
            dropped = torch.nonzero(box_fell, as_tuple=False).flatten()
            self._set_box_visibility(grabbed, visible=False)
            self._set_box_collision(grabbed, enabled=False)
            self._set_box_visibility(dropped, visible=True)
            self._set_box_collision(dropped, enabled=True)
            self._carry_held_boxes(anchor)

    # ── Active arm (Lane B): manual absolute-hold IK + clamp stack (ports drive_env_v2.py) ──
    def _ensure_arm_setup(self) -> None:
        """Lazy one-time setup for the active arm: IK controller, joint/body ids, Jacobian indexing,
        soft joint limits, shoulder centre, and the per-env target/smooth/freeze buffers."""
        if getattr(self, "_arm_ready", False):
            return
        from isaaclab.utils.math import subtract_frame_transforms
        robot: Articulation = self.scene["robot"]
        self._arm_joint_ids, _ = robot.find_joints(
            [f"panda_joint{i}" for i in range(1, 8)], preserve_order=True
        )
        self._arm_home = robot.data.default_joint_pos[:, self._arm_joint_ids].clone()
        self._ee_idx = robot.body_names.index("panda_hand")
        self._base_idx = robot.body_names.index("base_link")
        self._sh_idx = robot.body_names.index("panda_link0")
        # Jacobian columns: a welded floating base still reports a 6-col base block (see drive_env_v2).
        jac = robot.root_physx_view.get_jacobians()
        floating = jac.shape[-1] == robot.num_joints + 6
        self._jac_joint_ids = (
            [i + 6 for i in self._arm_joint_ids] if floating else list(self._arm_joint_ids)
        )
        self._ee_jacobi_idx = self._ee_idx if floating else self._ee_idx - 1
        # Soft joint limits (centred fraction of the hard USD range): the IK output is clamped to
        # these (the controller does not clamp its own), so an unreachable target can't drive joints
        # past range and oscillate.
        jlim = robot.data.joint_pos_limits[:, self._arm_joint_ids]      # (N,7,2) [min,max]
        j_mid = 0.5 * (jlim[..., 0] + jlim[..., 1])
        j_half = 0.5 * (jlim[..., 1] - jlim[..., 0])
        self._arm_jmin = j_mid - j_half * ARM_JOINT_MARGIN
        self._arm_jmax = j_mid + j_half * ARM_JOINT_MARGIN
        # IK controller: top-down pose, ABSOLUTE mode, DLS — matches drive_env_v2 --orient down.
        self._arm_ik = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls",
                ik_params={"lambda_val": ARM_IK_DAMP},
            ),
            num_envs=self.num_envs, device=self.device,
        )
        self._arm_ik.reset()
        # Per-env buffers. EE target + desired quat held in the base_link frame (the MOVING chassis),
        # re-expressed to the root frame each step so the target rides with the robot (#1268 fix).
        base_pos_w = robot.data.body_pos_w[:, self._base_idx]
        base_quat_w = robot.data.body_quat_w[:, self._base_idx]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            base_pos_w, base_quat_w,
            robot.data.body_pos_w[:, self._ee_idx], robot.data.body_quat_w[:, self._ee_idx],
        )
        sh_pos_b, _ = subtract_frame_transforms(
            base_pos_w, base_quat_w,
            robot.data.body_pos_w[:, self._sh_idx], robot.data.body_quat_w[:, self._sh_idx],
        )
        self._arm_center_b = sh_pos_b.clone()
        self._ee_target = ee_pos_b.clone()
        self._ee_quat_des = ee_quat_b.clone()
        self._arm_smooth = robot.data.joint_pos[:, self._arm_joint_ids].clone()
        self._arm_freeze_pose = self._arm_smooth.clone()
        self._arm_ready = True

    def _reanchor_arm(self, env_ids: torch.Tensor) -> None:
        """Re-anchor the active-arm buffers for reset envs to the current (spawn) pose so the next
        episode's first command doesn't jump. No-op before the first _ensure_arm_setup."""
        if not getattr(self, "_arm_ready", False) or env_ids.numel() == 0:
            return
        from isaaclab.utils.math import subtract_frame_transforms
        robot: Articulation = self.scene["robot"]
        base_pos_w = robot.data.body_pos_w[:, self._base_idx]
        base_quat_w = robot.data.body_quat_w[:, self._base_idx]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            base_pos_w, base_quat_w,
            robot.data.body_pos_w[:, self._ee_idx], robot.data.body_quat_w[:, self._ee_idx],
        )
        cur = robot.data.joint_pos[:, self._arm_joint_ids]
        self._ee_target[env_ids] = ee_pos_b[env_ids]
        self._ee_quat_des[env_ids] = ee_quat_b[env_ids]
        self._arm_smooth[env_ids] = cur[env_ids]
        self._arm_freeze_pose[env_ids] = cur[env_ids]

    def _drive_arm(self, ee_delta: torch.Tensor) -> None:
        """Active-arm control for stage >= 2 (ports scripts/drive_env_v2.py). Sets arm joint position
        targets only; the PD tracks them across the step's decimation substeps.

        NOT holding: accumulate the EE delta (base frame) into the absolute hold target; clamp (reach
        box + leash + radial); solve top-down pose IK re-expressed base->root (the #1268 ride-with-base
        fix); clamp joints to soft limits; EXP-smooth; cap the per-step joint delta. HOLDING: hold the
        grab-pose snapshot — the box is FixedJoint-welded to base_link at grab, so the hand must stay.
        """
        from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms
        self._ensure_arm_setup()
        robot: Articulation = self.scene["robot"]
        held = self.holding
        free = ~held
        if bool(free.any()):                                   # held envs keep their frozen target
            self._ee_target[free] = self._ee_target[free] + ee_delta[free]
        self._ee_target[..., 0] = self._ee_target[..., 0].clamp_min(ARM_REACH_X_BACK)
        self._ee_target[..., 2] = self._ee_target[..., 2].clamp(ARM_REACH_Z_MIN, ARM_REACH_Z_MAX)
        base_pos_w = robot.data.body_pos_w[:, self._base_idx]
        base_quat_w = robot.data.body_quat_w[:, self._base_idx]
        ee_now_b, _ = subtract_frame_transforms(
            base_pos_w, base_quat_w,
            robot.data.body_pos_w[:, self._ee_idx], robot.data.body_quat_w[:, self._ee_idx],
        )
        if ARM_LEAD_M > 0.0:                                   # leash: target leads live EE by <= lead
            lead = self._ee_target - ee_now_b
            n = torch.norm(lead, dim=-1, keepdim=True)
            self._ee_target = ee_now_b + lead * (n.clamp(max=ARM_LEAD_M) / n.clamp_min(1e-6))
        if ARM_REACH_R > 0.0:                                  # radial workspace clamp around shoulder
            v = self._ee_target - self._arm_center_b
            r = torch.norm(v, dim=-1, keepdim=True).clamp_min(1e-6)
            self._ee_target = self._arm_center_b + v * (r.clamp(ARM_REACH_RMIN, ARM_REACH_R) / r)
        # Absolute pose IK: base-relative target -> world -> root frame (rides with the base, #1268).
        jacobian = robot.root_physx_view.get_jacobians()[:, self._ee_jacobi_idx, :, self._jac_joint_ids]
        root_w = robot.data.root_pose_w
        ee_pos_r, ee_quat_r = subtract_frame_transforms(
            root_w[:, 0:3], root_w[:, 3:7],
            robot.data.body_pos_w[:, self._ee_idx], robot.data.body_quat_w[:, self._ee_idx],
        )
        tgt_pos_w, tgt_quat_w = combine_frame_transforms(
            base_pos_w, base_quat_w, self._ee_target, self._ee_quat_des
        )
        tgt_pos_r, tgt_quat_r = subtract_frame_transforms(
            root_w[:, 0:3], root_w[:, 3:7], tgt_pos_w, tgt_quat_w
        )
        joint_pos = robot.data.joint_pos[:, self._arm_joint_ids]
        self._arm_ik.set_command(torch.cat([tgt_pos_r, tgt_quat_r], dim=-1))
        ik_targets = self._arm_ik.compute(ee_pos_r, ee_quat_r, jacobian, joint_pos)
        ik_targets = torch.clamp(ik_targets, self._arm_jmin, self._arm_jmax)
        self._arm_smooth = ARM_SMOOTH_BETA * ik_targets + (1.0 - ARM_SMOOTH_BETA) * self._arm_smooth
        q_cur = robot.data.joint_pos[:, self._arm_joint_ids]
        targets = q_cur + torch.clamp(self._arm_smooth - q_cur, -ARM_MAX_JOINT_STEP, ARM_MAX_JOINT_STEP)
        if bool(held.any()):                                   # hold grab pose; re-sync target for clean release
            targets[held] = self._arm_freeze_pose[held]
            self._arm_smooth[held] = self._arm_freeze_pose[held]
            self._ee_target[held] = ee_now_b[held]
        robot.set_joint_position_target(targets, joint_ids=self._arm_joint_ids)

    def _freeze_held_arm(self) -> None:
        """Hard-pin held envs' arm to the grab-pose snapshot after the sim step (stability: the box is
        welded to base_link, so the hand must not drift). Active-arm carry analogue of _freeze_arm."""
        if not getattr(self, "_arm_ready", False) or not bool(self.holding.any()):
            return
        robot: Articulation = self.scene["robot"]
        held = torch.nonzero(self.holding, as_tuple=False).flatten()
        pose = self._arm_freeze_pose[held]
        robot.write_joint_state_to_sim(
            pose, torch.zeros_like(pose), joint_ids=self._arm_joint_ids, env_ids=held
        )

    def _freeze_arm(self) -> None:
        """Kinematically pin the 7 arm joints to their home pose each step (rigid scenery).

        The arm is driven by relative-mode IK with a zero EE delta, which only commands "stay at the
        CURRENT pose" — base acceleration then nudges the arm and it drifts/flails with no restoring
        force back to home. Since the arm never actuates here (grasp is magnetic, box rides on
        base_link), overwrite the arm joint state to the home config + zero velocity so it stays rigid
        relative to the chassis. See bugs_errors/2026-06-21_frozen-arm-snags-racks.md.
        """
        robot: Articulation = self.scene["robot"]
        if not hasattr(self, "_arm_joint_ids"):
            self._arm_joint_ids, _ = robot.find_joints(
                [f"panda_joint{i}" for i in range(1, 8)], preserve_order=True
            )
            self._arm_home = robot.data.default_joint_pos[:, self._arm_joint_ids].clone()
        zeros = torch.zeros_like(self._arm_home)
        robot.write_joint_state_to_sim(self._arm_home, zeros, joint_ids=self._arm_joint_ids)

    def _update_stuck(self) -> None:
        """Count consecutive idle (no-translation) control steps for the stuck_timeout termination."""
        robot: Articulation = self.scene["robot"]
        bidx = robot.body_names.index("base_link")
        xy = robot.data.body_pos_w[:, bidx, :2]
        moved = torch.norm(xy - self._prev_base_xy, dim=-1)
        self._prev_base_xy = xy.clone()
        idle = moved < STUCK_MOVE_EPS_M
        self._stuck_steps = torch.where(
            idle, self._stuck_steps + 1.0, torch.zeros_like(self._stuck_steps)
        )

    def _update_carry_progress(self) -> None:
        """Count consecutive HOLDING steps where the box gets no closer to its goal zone (backing up
        / dawdling). Resets on progress (> CARRY_PROGRESS_EPS_M closer) or when not holding. Drives
        carry_regress_penalty so the robot doesn't reverse forever on the way to the finish zone."""
        cur = torch.norm(self.box_pos[:, :2] - self.goal_pos[:, :2], dim=-1)
        progressed = cur < (self._prev_carry_dist - CARRY_PROGRESS_EPS_M)
        stalling = self.holding & (~progressed)   # only while carrying; approach phase never counts
        self._carry_regress_steps = torch.where(
            stalling, self._carry_regress_steps + 1.0, torch.zeros_like(self._carry_regress_steps)
        )
        self._prev_carry_dist = cur

    def _update_checkpoint(self) -> None:
        """Snapshot the sim state (env 0) while carrying — at grasp and each closer CHECKPOINT_RING_M
        ring to the goal. Restored on a failure reset so the next episode resumes near the goal.
        num_envs==1 only (reuses recording.sim_state, which is env-0 scoped)."""
        if self.num_envs != 1 or not bool(self.holding[0]):
            return
        dist = float(torch.norm(self.box_pos[0, :2] - self.goal_pos[0, :2]))
        ring = int(dist // CHECKPOINT_RING_M)
        if bool(self._ckpt_valid[0]) and ring >= int(self._ckpt_ring[0]):
            return   # no new (closer) ring reached
        from recording.sim_state import capture_sim_state
        self._checkpoint = capture_sim_state(self)
        self._ckpt_valid[0] = True
        self._ckpt_ring[0] = ring

    def _restore_checkpoint_env0(self, saved_target: tuple) -> None:
        """Overwrite env-0 with the saved checkpoint AFTER the normal fresh reset ran (so a failure
        here degrades to a valid fresh spawn). Reverts the resampled target to the carried box,
        restores joints+box+holding, re-welds the box, and re-arms the active-arm freeze pose."""
        from recording.sim_state import restore_sim_state
        name, cat, goal, gid = saved_target
        self.target_box_name[0] = name
        self.box_cat_idx[0] = cat
        self.goal_pos[0] = goal
        self.goal_id_buf[0] = gid
        restore_sim_state(self, self._checkpoint)        # joints (base+arm) + box pose + holding
        self._ever_grasped[0] = True
        ids0 = torch.tensor([0], device=self.device)
        if CARRY_MODE == "physics":                       # re-weld the carried box at its restored pose
            box = self.scene[name]
            self._attach_boxes(ids0, box.data.root_pos_w[0:1])
            self._set_box_collision(ids0, enabled=False)
        else:
            self._set_box_visibility(ids0, visible=False)
            self._set_box_collision(ids0, enabled=False)
        if getattr(self, "_arm_ready", False):            # active-arm carry freezes at the grab pose
            self._arm_freeze_pose[0] = self._checkpoint["joint_pos"][0, self._arm_joint_ids]

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

    # ── Physics grasp (CARRY_MODE == "physics") ──────────────────────────────────────
    def _ensure_stage(self):
        """Lazily fetch + cache the USD stage (only needed for physics grasp)."""
        if self._usd_stage is None:
            import omni.usd
            self._usd_stage = omni.usd.get_context().get_stage()
        return self._usd_stage

    def _disable_arm_collisions(self) -> None:
        """Turn off collision on every Franka arm/hand link so the frozen arm can't snag racks.

        The grasp is magnetic (proximity, not physical contact) and the carried box is welded to
        base_link, so the arm colliders serve no purpose and only catch on rack frames mid-carry.
        Base/chassis colliders (path has no '/panda_') are left intact so the base still blocks
        against racks and avoidance still works.
        """
        from pxr import Usd, UsdPhysics
        stage = self._ensure_stage()
        for e in range(self.num_envs):
            root = stage.GetPrimAtPath(f"/World/envs/env_{e}/Robot")
            if not root.IsValid():
                continue
            for prim in Usd.PrimRange(root):
                if "/panda_" in prim.GetPath().pathString and prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)

    def _box_prim_path(self, e: int) -> str:
        """Resolved rigid-body prim path of env e's target box."""
        return f"/World/envs/env_{e}/{self.target_box_name[e]}"

    def _base_link_path(self, e: int) -> str:
        """Resolved chassis (base_link) prim path — the carry weld parent (box rides in front)."""
        return f"/World/envs/env_{e}/Robot/base_link"

    def _set_box_visibility(self, env_ids: torch.Tensor, visible: bool) -> None:
        """Show/hide each env's target box prim (hidden-kinematic carry: box not rendered while held)."""
        ids = env_ids.tolist() if hasattr(env_ids, "tolist") else list(env_ids)
        if not ids:
            return
        from pxr import UsdGeom
        stage = self._ensure_stage()
        for e in ids:
            img = UsdGeom.Imageable(stage.GetPrimAtPath(self._box_prim_path(e)))
            (img.MakeVisible if visible else img.MakeInvisible)()

    def _set_box_collision(self, env_ids: torch.Tensor, enabled: bool) -> None:
        """Enable/disable collision on each env's target box (off while carried so it can't hit racks)."""
        ids = env_ids.tolist() if hasattr(env_ids, "tolist") else list(env_ids)
        if not ids:
            return
        from pxr import Usd, UsdPhysics
        stage = self._ensure_stage()
        for e in ids:
            root = stage.GetPrimAtPath(self._box_prim_path(e))
            for prim in Usd.PrimRange(root):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(enabled)

    def _grip_anchor_world(self) -> torch.Tensor:
        """World-frame carry point (N,3) = the panda_hand (grabber) position.

        The held box is snapped here and welded to the chassis, so it sits AT the gripper (not a
        floating offset in front of the base). Arm is frozen, so the hand→base offset is constant →
        the box rides with the robot at the grabber. (GRIP_FWD/GRIP_UP kept for the kinematic mode.)
        """
        robot: Articulation = self.scene["robot"]
        hand = robot.body_names.index("panda_hand")
        return robot.data.body_pos_w[:, hand].clone()           # (N,3) hand/grabber world pos

    def _hand_path(self, e: int) -> str | None:
        """Resolved panda_hand LINK prim path for env e (cached; traverses Robot subtree once)."""
        if e not in self._hand_prim_path:
            from env.attach import find_descendant_path
            self._hand_prim_path[e] = find_descendant_path(
                self._ensure_stage(), f"/World/envs/env_{e}/Robot", "panda_hand"
            )
        return self._hand_prim_path[e]

    def _snap_boxes_to_ee(self, env_ids: torch.Tensor, ee_world: torch.Tensor) -> None:
        """Teleport each given env's target box to the EE (used just before a physics weld)."""
        for e in env_ids.tolist():
            box = self.scene[self.target_box_name[e]]
            state = box.data.root_state_w[e:e + 1].clone()
            state[:, 0:3] = ee_world[e:e + 1]
            state[:, 3:7] = 0.0
            state[:, 3] = 1.0      # upright (qw=1) so the welded relative rotation is deterministic
            state[:, 7:13] = 0.0
            box.write_root_state_to_sim(state, env_ids=torch.tensor([e], device=self.device))

    def _attach_boxes(self, env_ids: torch.Tensor, anchor: torch.Tensor) -> None:
        """FixedJoint-weld each env's target box to base_link at the carry anchor (physics carry).

        Authors the joint's body0 anchor = box pose in the base_link frame, so PhysX holds the box
        exactly where it was snapped (in front of + above the chassis) instead of yanking it onto
        the base_link origin. `anchor` (N,3) = the world carry point the box was just snapped to.
        """
        from env.attach import attach_box
        from isaaclab.utils.math import subtract_frame_transforms
        stage = self._ensure_stage()
        robot: Articulation = self.scene["robot"]
        base = robot.body_names.index("base_link")
        p_b = robot.data.body_pos_w[:, base]                       # (N,3) chassis world pos
        q_b = robot.data.body_quat_w[:, base]                      # (N,4) chassis world quat
        q_box = torch.zeros_like(q_b); q_box[:, 0] = 1.0           # box upright (matches snap)
        # Box pose expressed in the base_link frame = the joint's body0 local anchor.
        p_rel, q_rel = subtract_frame_transforms(p_b, q_b, anchor, q_box)
        for e in env_ids.tolist():
            attach_box(
                stage, self._base_link_path(e), self._box_prim_path(e),
                local_pos0=tuple(p_rel[e].tolist()), local_rot0=tuple(q_rel[e].tolist()),
            )

    def _detach_boxes(self, env_ids: torch.Tensor) -> None:
        """Remove any grasp FixedJoint on each env's target box."""
        from env.attach import detach_box
        stage = self._ensure_stage()
        for e in env_ids.tolist():
            detach_box(stage, self._box_prim_path(e))

    def _apply_physics_grasp(self, newly: torch.Tensor, released: torch.Tensor,
                             ee_world: torch.Tensor) -> None:
        """Weld newly-grasped boxes to the hand; unweld released ones (replaces kinematic carry)."""
        newly_ids = torch.nonzero(newly, as_tuple=False).flatten()
        if newly_ids.numel():
            self._snap_boxes_to_ee(newly_ids, ee_world)  # clean relative transform at weld time
            self._attach_boxes(newly_ids, ee_world)
            # Box rides 0.6 m in front of the chassis; with collision ON it rams the rack the box was
            # just lifted off and pins the base. Delivery is position-based (box_pos vs zone), so the
            # carried box needs no collision — turn it off while held (re-enabled on release).
            self._set_box_collision(newly_ids, enabled=False)
        rel_ids = torch.nonzero(released, as_tuple=False).flatten()
        if rel_ids.numel():
            self._detach_boxes(rel_ids)
            self._set_box_collision(rel_ids, enabled=True)

    # ── Per-stage reset (Stage 1 pre-grasp, Stage 2 spawn-near-box) ───────────────────
    def _apply_stage_reset(self, env_ids: torch.Tensor) -> None:
        """Apply stage-2 spawn-near-box and stage-1 pre-grasp overrides after the normal reset."""
        from env.curriculum import stage_is_pregrasped, stage_is_spawn_near_box
        if stage_is_spawn_near_box(self.stage):
            self._spawn_base_near_box(env_ids)
        if stage_is_pregrasped(self.stage):
            self._pregrasp_box(env_ids)

    def _spawn_base_near_box(self, env_ids: torch.Tensor) -> None:
        """Stage 2: place the chassis within Franka reach of the target box, facing it.

        Overrides the receiving-north spawn from EventCfg.reset_robot. Writes the robot ROOT pose
        (the welded fixed-base anchor) and zeros the dummy base joints so base_link lands at the
        root pose. box_pos must already be refreshed (called after _refresh_target_box_pos).
        VERIFY in sim: confirm base_link lands beside the box (scripts/tune_arm.py / drive_env.py).
        """
        from env.curriculum import spawn_pose_near_box
        robot: Articulation = self.scene["robot"]
        root = robot.data.root_state_w.clone()  # (N,13) world frame: pos3+quat4+linvel3+angvel3
        for e in env_ids.tolist():
            bx, by = float(self.box_pos[e, 0]), float(self.box_pos[e, 1])
            # Standoff must CLEAR the rack: box sits at the rack centre, the rack/shelf-deck footprint
            # is ~0.9 m deep (SHELF_DECK_SIZE) and the Ridgeback base is ~0.96 m long (~0.5 m half),
            # so a base centre nearer than ~0.85 m spawns INSIDE the rack and the sim explodes (see
            # bugs_errors/2026-06-21_stage2-spawn-inside-rack.md). Spawn in the open aisle and let the
            # demo controller / policy drive the last ~0.4 m in for the magnetic grab. Heavy boxes are
            # bigger → slightly larger standoff. Tune in sim with scripts/demo_pickup.py.
            # Optional (small, heavy) override set by the demo's tuning config; default clears rack.
            small_h = getattr(self, "_spawn_standoff", None)
            heavy = self._box_size[self.target_box_name[e]] > 0.4
            if small_h is not None:
                standoff = small_h[1] if heavy else small_h[0]
            else:
                standoff = 1.1 if heavy else 1.0
            base_x, base_y, yaw = spawn_pose_near_box((bx, by), standoff=standoff)
            origin = self.scene.env_origins[e]
            root[e, 0] = origin[0] + base_x
            root[e, 1] = origin[1] + base_y
            half = yaw * 0.5
            root[e, 3] = math.cos(half)   # qw
            root[e, 4] = 0.0              # qx
            root[e, 5] = 0.0              # qy
            root[e, 6] = math.sin(half)   # qz (yaw about world z)
            root[e, 7:13] = 0.0
        robot.write_root_state_to_sim(root, env_ids=env_ids)
        base_ids, _ = robot.find_joints(
            ["dummy_base_prismatic_x_joint", "dummy_base_prismatic_y_joint",
             "dummy_base_revolute_z_joint"], preserve_order=True,
        )
        zeros = torch.zeros(env_ids.numel(), len(base_ids), device=self.device)
        robot.write_joint_state_to_sim(zeros, zeros, joint_ids=base_ids, env_ids=env_ids)

    def _pregrasp_box(self, env_ids: torch.Tensor) -> None:
        """Stage 1: start with the target box already held (snap to EE + weld + holding=True)."""
        anchor = self._grip_anchor_world()
        self._snap_boxes_to_ee(env_ids, anchor)
        if CARRY_MODE == "physics":
            self._attach_boxes(env_ids, anchor)
        self.holding[env_ids] = True


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

    def __init__(self, cfg: WarehouseEnvCfg | None = None, render_mode: str | None = None,
                 arm_active: bool = False):
        """Build underlying RL env and Gym-style spaces.

        Arm control is STAGE-GATED (Lane B, 2026-06-22): stage >= 2 (grasp/full/anneal) drives the
        arm via WarehouseRLEnv._drive_arm (absolute-hold EE target + clamp stack, ported from the
        verified drive_env_v2.py); stage 1 (pre-grasped nav) keeps the arm frozen at home. Set
        arm_active=True to FORCE the active arm regardless of stage (e.g. drive_env teleop). The
        external (6,) action and obs dict are unchanged either way.
        """
        self.cfg = cfg if cfg is not None else WarehouseEnvCfg()
        self.arm_active = arm_active
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
        from env.curriculum import STAGE_GRASP
        base2, ee3, grip1 = split_action(action)
        base3 = self._base_cmd(base2)                       # (N,3) base joint velocities
        internal = torch.cat([base3, grip1], dim=-1)        # (N,4) base(3)+gripper(1); arm driven manually
        # Lane B (2026-06-22): ACTIVE arm for stage >= 2 (grasp needs it) or an explicit arm_active
        # override; FROZEN for stage 1 (pre-grasped nav — isolates carry from arm motion). _drive_arm
        # sets the arm joint targets BEFORE the step (the PD tracks them across the decimation substeps);
        # _freeze_held_arm hard-pins held envs after (box welded to base_link, hand must not drift).
        active_arm = self.arm_active or self._env.stage >= STAGE_GRASP
        if active_arm:
            self._env._drive_arm(ee3)
        obs, reward, terminated, truncated, info = self._env.step(internal)
        if active_arm:
            self._env._freeze_held_arm()                    # carry: pin held envs' arm to the grab pose
        else:
            self._env._freeze_arm()                         # frozen training: pin arm to home
        self._env.update_grasp()                            # holding + grasp/drop events, carry box
        self._env._update_stuck()                           # idle detector for stuck_timeout reset
        self._env._update_carry_progress()                  # backing-up-from-zone detector
        self._env._update_checkpoint()                      # snapshot for reset-to-checkpoint
        # Surface the one-shot task events so consumers (recorder summary, P3 buffer) don't have to
        # reach into ._env. info gains 3 (N,) bool tensors; the (obs,r,term,trunc,info) shape is unchanged.
        if isinstance(info, dict):
            info["grasp_event"] = self._env.grasp_event
            info["drop_event"] = self._env.drop_event
            info["deliver_event"] = self._env.deliver_event
            # Reward tracking: weighted per-term breakdown (env 0) + total, so any trainer/recorder
            # can log WHICH term drives the step. Reuses env.reward_debug (also used by drive_env).
            from env.reward_debug import reward_breakdown
            terms = reward_breakdown(self._env)
            info["reward_terms"] = terms
            info["reward_total"] = sum(v for v in terms.values() if v == v)  # skip NaN
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
