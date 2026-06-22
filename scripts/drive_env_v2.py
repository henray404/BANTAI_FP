# drive_env_v2.py — LIGHTWEIGHT gripper/grasp tuning teleop (1 rack + 1 box).
#
# Why a v2: drive_env.py runs the FULL task env (18 racks, 54 shelf decks w/ materials, 18
# boxes, props, walls, zones, onboard camera) and rendered windowed it crawls (~3 fps on the
# 8GB RTX 5050 — render-bound, NOT physics). This strips the scene to the minimum needed to
# tune the GRASP: robot + 1 rack + 1 box. No onboard camera (biggest render save — gripper
# tuning needs numbers, not pixels), no decks/props/walls/zones (kills the material-node load
# that the Blackwell SDP path hates). Borrows drive_robot.py's proven absolute-pose IK loop.
#
# FPS recommendations baked in (see CLAUDE.md audit 2026-06-22):
#   - camera OFF             : the single biggest render cost is gone.
#   - --render_every 4       : render at ~50 Hz (every 4 physics steps) not 200 Hz -> 4x fewer
#                              render calls. Raise it for more fps, lower for smoother view.
#   - minimal asset count    : ~3 prims vs ~150 -> light geometry + few material nodes.
#   ALSO (free, do it yourself): shrink the Isaac Sim window (fewer pixels ray-traced) and
#   check `nvidia-smi` — if VRAM is full the render stalls; close other GPU apps.
#
# AppLauncher MUST be created here, before any isaaclab imports from env/.
# See bugs_errors/2026-05-15_double-applaunch-crash.md.
#
# Usage:
#   conda activate isaaclab
#   python scripts/drive_env_v2.py                       # 21cm box, render every 4 steps
#   python scripts/drive_env_v2.py --box_size 0.52       # heavy (52cm) box
#   python scripts/drive_env_v2.py --render_every 8      # faster (choppier view)
#   python scripts/drive_env_v2.py --grip_radius 0.30    # try a looser grasp radius
#
# Keys (window must be focused):
#   BASE: arrows = fwd/back + strafe   Z/X = yaw +/-
#   ARM : W/S = EE +x/-x   A/D = EE +y/-y   Q/E = EE +z/-z   K = toggle gripper
# Telemetry (~5 Hz): ee<->box SURFACE distance, gripper open/closed, and whether the env's
#   grasp_success() WOULD fire — the exact signal to tune GRIP_RADIUS_M / home pose against.

"""Lightweight 1-rack/1-box teleop for tuning the Ridgeback-Franka grasp + gripper."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# ── CLI + AppLauncher ─────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Lightweight grasp-tuning teleop (1 rack, 1 box)")
parser.add_argument("--lin", type=float, default=1.5, help="forward/back speed sensitivity (m/s)")
parser.add_argument("--ang", type=float, default=1.5, help="yaw speed sensitivity (rad/s)")
parser.add_argument("--strafe", type=float, default=0.0, help="lateral speed (0 = no strafe)")
parser.add_argument("--ee_sens", type=float, default=0.002,
                    help="EE position delta per physics step while an arm key is held (m). "
                         "Smaller = calmer arm (less overshoot/flail). 0.002 ~= 0.4 m/s reach.")
parser.add_argument("--arm_smooth", type=float, default=0.2,
                    help="EXP low-pass on the arm joint targets in [0,1]: 1.0 = none (raw IK, "
                         "twitchy), 0.2 = heavy smoothing (calm, slight lag). Tames flail from "
                         "fast keys + base-motion transients. Same control should port to training.")
parser.add_argument("--orient", choices=["down", "free"], default="down",
                    help="Gripper orientation. 'down' (default) = pose IK holds the spawn top-down "
                         "hand orientation, DAMPED (dls lambda 0.1) so fingers stay pointing at the "
                         "box without the z-collapse flip. 'free' = position-only IK (orientation "
                         "unconstrained, hand twists 'di dalam'); fall back to it if 'down' still "
                         "collapses on descent.")
parser.add_argument("--ik_damp", type=float, default=0.1,
                    help="DLS damping (lambda) for 'down'/pose IK. Higher = smoother near singularities "
                         "but SUPPRESSES horizontal (xy) motion when the arm is extended ('cuma bisa z'); "
                         "lower = freer xy but the z-collapse flip can return. 0.1 default (no flip). The "
                         "real fix for 'cuma bisa z' is posture, not lambda: drive the base close so the "
                         "arm works mid-workspace (--box_dist 0.6). ('free' mode uses 0.01.)")
parser.add_argument("--max_joint_step", type=float, default=0.04,
                    help="Hard cap on |joint target - current| per physics step (rad). Stops a "
                         "near-singular IK solve from flipping the arm in one step ('langsung jatuh'). "
                         "~0.04 rad/step @ dt=0.005s ~= 8 rad/s ceiling; normal teleop is well under it.")
parser.add_argument("--freeze_on_grip", action="store_true",
                    help="HARD freeze: once the box is grabbed, lock the arm at the grab pose and ignore "
                         "arm keys — only the base drives (carry to the zone without the arm wandering "
                         "into a rack). Release (K) unlocks it. Without this flag the arm holds the grab "
                         "pose but stays key-movable (soft freeze).")
parser.add_argument("--reach_x_back", type=float, default=0.0,
                    help="Min EE-target x in the BASE frame (m): backward reach (S key, -x) is clamped "
                         "here so the gripper can't drive back into the base/rack ('mundur nabrak'). "
                         "Watch tgt(b) x in the telemetry to find the nabrak point, then set this. "
                         "Forward (+x) stays unbounded.")
parser.add_argument("--reach_z_max", type=float, default=1.1,
                    help="Max EE-target z in the BASE frame (m): caps how HIGH the gripper rises ('gak "
                         "bisa tinggi-tinggi banget'). Past it, up-input (Q) is ignored. Tune from "
                         "tgt(b) z in the telemetry; calib reachable top ~1.41.")
parser.add_argument("--reach_z_min", type=float, default=0.05,
                    help="Min EE-target z in the BASE frame (m): floor on how LOW the gripper drops "
                         "(down-input E ignored past it). Keep low enough to reach the floor box "
                         "(box center = box_size/2). Calib reachable floor ~0.36.")
parser.add_argument("--reach_xy", type=float, default=0.3,
                    help="EE target clamp half-size on X/Y (m) around the spawn pose.")
parser.add_argument("--reach_z", type=float, default=0.18,
                    help="EE target clamp half-size on Z (m). Tighter than XY: vertical motion "
                         "hits the Franka elbow (joint4) / wrist (joint6) limits fastest.")
parser.add_argument("--lead", type=float, default=0.08,
                    help="Max distance (m) the commanded EE target may LEAD the actual EE. Stops "
                         "the target outrunning reach so the arm never strains at an unreachable "
                         "point ('maksa di limit'). Smaller = tighter follow.")
parser.add_argument("--joint_margin", type=float, default=0.95,
                    help="Soft joint-limit fraction in [0,1]: clamp arm joints to this fraction of "
                         "their hard USD range, centred (0.95 = stop ~5%% short of hard stops -> no "
                         "chatter). Hard limits still feed the calibration margin metric.")
parser.add_argument("--reach_r", type=float, default=0.0,
                    help="Radial workspace clamp: max |EE - shoulder| (m), couples x/y/z. 0 = OFF "
                         "(default — free teleop for grasp exploration). The clamp is STATEFUL (snaps "
                         "ee_target onto the r_max shell every step), and the fitted r_max=0.95 sits in "
                         "the LOW-manipulability near-singular zone (calib healthy band 0.65-0.90, spawn "
                         "~0.934) -> 'dinding' that jitters. Re-enable with e.g. --reach_r 0.90 (top of "
                         "the healthy band, not the 0.95 jitter edge) if you want an outer guard.")
parser.add_argument("--reach_rmin", type=float, default=0.50,
                    help="Radial workspace clamp: min |EE - shoulder| (m) — avoids folding into the "
                         "body / inner singularity. Fitted: swept data floor ~0.62, 0.50 leaves room.")
parser.add_argument("--calib", type=str, nargs="?", default="", const="calib/arm_envelope.csv",
                    help="Calibration mode: write a per-step CSV and FREE the envelope (leash + "
                         "radial + box clamps off) so you can sweep the reachable space. Logs ee "
                         "xyz, radius, joint margin, pose error, and manipulability (singularity "
                         "sensor). Bare `--calib` -> calib/arm_envelope.csv; `--calib path.csv` -> "
                         "that path (relative to project root). Omit = normal teleop.")
parser.add_argument("--box_size", type=float, default=0.21, choices=[0.21, 0.32, 0.52],
                    help="box edge length (m): 0.21 fragile / 0.32 regular / 0.52 heavy.")
parser.add_argument("--box_dist", type=float, default=1.2,
                    help="box distance in front of the robot spawn (m, along +x).")
parser.add_argument("--grip_radius", type=float, default=None,
                    help="ee<->box surface radius to test the grasp at (m). "
                         "Default = env's shipped GRIP_RADIUS_M.")
parser.add_argument("--no_rack", action="store_true", help="skip the rack (box only — lightest).")
parser.add_argument("--render_every", type=int, default=4,
                    help="render once per N physics steps (higher = more fps, choppier). dt=0.005s.")
# ── Scripted magnetic pickup demo (port of scripts/demo_pickup.py onto this lightweight scene) ──
parser.add_argument("--pickup", action="store_true",
                    help="SCRIPTED pickup demo: freeze arm, auto-drive base to the box -> magnetic "
                         "grasp (gripper held closed) -> carry to the delivery zone -> deliver. Same "
                         "magnetic grasp as training (env.grasp.grasp_success), no keyboard. Use "
                         "--box_size 0.52 (heavy): the frozen hand sits ~0.59 m up, so only the 52 cm "
                         "box is within GRIP_RADIUS on the floor.")
parser.add_argument("--zone_x", type=float, default=0.0, help="delivery zone x (m), env-local.")
parser.add_argument("--zone_y", type=float, default=-3.0, help="delivery zone y (m), env-local.")
parser.add_argument("--episodes", type=int, default=1, help="pickup episodes to run (re-place box each).")
parser.add_argument("--max_steps", type=int, default=1500, help="physics-step cap per pickup episode.")
# Auto-calibration defaults: if calib/arm_calib.yaml exists (written by calibrate_arm.py --auto),
# use its fitted reach_r/reach_rmin as the radial-clamp defaults so the robot's calibration is used
# without flags (a CLI flag still overrides). Home pose is applied by warehouse_scene separately.
_ARM_CALIB = Path(__file__).resolve().parents[1] / "calib" / "arm_calib.yaml"
if _ARM_CALIB.exists():
    try:
        import yaml as _yaml
        _ac = (_yaml.safe_load(_ARM_CALIB.read_text(encoding="utf-8")) or {}).get("arm_calib", {})
    except ImportError:
        import ruamel.yaml as _ryaml
        _ac = (_ryaml.YAML(typ="safe").load(_ARM_CALIB.read_text(encoding="utf-8")) or {}).get("arm_calib", {})
    parser.set_defaults(**{k: float(_ac[k]) for k in ("reach_r", "reach_rmin") if k in _ac})
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# No --enable_cameras: this scene mounts NO onboard camera, so the SDP graph never inits.

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Disable the PhysX Support UI extension (spams "invalid null prim" on prim rewrites; GUI-only).
import omni.kit.app  # noqa: E402
omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
    "omni.physx.supportui", False
)

# ── Imports (after AppLauncher) ───────────────────────────────────────
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg, Se3Keyboard, Se3KeyboardCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from env.warehouse_scene import RIDGEBACK_FRANKA_CFG, _item_cfg, _rack_cfg  # noqa: E402
from env.grasp import GRIP_RADIUS_M, grasp_success  # noqa: E402

BASE_JOINTS = [
    "dummy_base_prismatic_x_joint",
    "dummy_base_prismatic_y_joint",
    "dummy_base_revolute_z_joint",
]
ARM_JOINT_RE = "panda_joint.*"
FINGER_JOINT_RE = "panda_finger_joint.*"
EE_BODY = "panda_hand"
FINGER_LEN = 0.10   # panda_hand origin -> fingertip (m); carried box hangs this + box_half below it
CARRY_GAP_M = 0.05  # gap between hand origin and carried box near-face when floating it in front
GRIP_OPEN, GRIP_CLOSE = 0.035, 0.0
FINGER_CLOSED_THRESH = 0.0175  # < half of open (0.035) — matches env.update_grasp
BOX_MASS = {0.21: 2.0, 0.32: 6.0, 0.52: 12.0}


@configclass
class MiniSceneCfg(InteractiveSceneCfg):
    """Minimal scene: ground + light + robot only. Rack/box added at runtime (size from CLI)."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8, dynamic_friction=0.6, restitution=0.0
            ),
        ),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 0.95)),
    )
    robot = RIDGEBACK_FRANKA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _build_scene() -> tuple[SimulationContext, InteractiveScene, float]:
    """Create sim + a 1-rack/1-box scene. Returns (sim, scene, box_half)."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, render_interval=max(1, args_cli.render_every))
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(-3.0, -2.5, 2.5), target=(args_cli.box_dist, 0.0, 0.3))
    scene_cfg = MiniSceneCfg(num_envs=1, env_spacing=8.0)
    size = float(args_cli.box_size)
    # Box on the FLOOR in front of the robot (matches the real training env — env_config places the
    # graspable boxes on the floor in front of racks). The Ridgeback base is long (~0.96 m), so a box
    # ON a shelf forces the base INTO the rack to reach it -> wedges in the open frame. Floor-in-front
    # keeps the robot in front of the rack, grasping without touching it.
    scene_cfg.box = _item_cfg("box", size, BOX_MASS.get(size, 2.0), (args_cli.box_dist, 0.0, size / 2.0))
    if not args_cli.no_rack:
        # Rack BEHIND the box (backdrop), clear of the robot's approach so the base can't wedge into it.
        scene_cfg.rack_0 = _rack_cfg(0, (args_cli.box_dist + 0.7, 0.0, 0.0))
    scene = InteractiveScene(scene_cfg)
    return sim, scene, size / 2.0


def _ee_pose_base(robot, ee_body_idx) -> tuple[torch.Tensor, torch.Tensor]:
    """Current EE pose (pos (1,3), quat (1,4)) in the articulation root/base frame."""
    ee_w = robot.data.body_pose_w[:, ee_body_idx]
    root_w = robot.data.root_pose_w
    return subtract_frame_transforms(
        root_w[:, 0:3], root_w[:, 3:7], ee_w[:, 0:3], ee_w[:, 3:7]
    )


def _arm_ik_targets(robot, arm_ik, arm_ids, jac_joint_ids, ee_body_idx, ee_jacobi_idx,
                    base_link_idx, ee_target_b, ee_quat_b) -> torch.Tensor:
    """Solve absolute pose IK to an EE target held in the CHASSIS (base_link) frame -> arm joints.

    FIX for the mobile base: the target is given relative to base_link and re-expressed each step
    into the articulation-root frame (the welded `world` link, fixed at origin) that the Jacobian +
    controller use. Holding a ROOT/world-fixed target made the arm reach back toward origin while
    the chassis drove away -> drag/flail (same root-vs-base_link trap as IsaacLab #1268). Expressing
    it base-relative makes the arm RIDE WITH the chassis instead.
    """
    jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, jac_joint_ids]
    ee_pos_r, ee_quat_r = _ee_pose_base(robot, ee_body_idx)              # current EE, root frame
    base_pos_w = robot.data.body_pos_w[:, base_link_idx]
    base_quat_w = robot.data.body_quat_w[:, base_link_idx]
    tgt_pos_w, tgt_quat_w = combine_frame_transforms(base_pos_w, base_quat_w, ee_target_b, ee_quat_b)
    root_w = robot.data.root_pose_w
    tgt_pos_r, tgt_quat_r = subtract_frame_transforms(
        root_w[:, 0:3], root_w[:, 3:7], tgt_pos_w, tgt_quat_w
    )
    joint_pos = robot.data.joint_pos[:, arm_ids]
    # 'pose' = track position + the (top-down) orientation; HEAVY-damped so it tilts, not flips, near
    # the wrist limit. 'position' = orientation FREE (no collapse, but the hand twists 'di dalam').
    if arm_ik.cfg.command_type == "pose":
        arm_ik.set_command(torch.cat([tgt_pos_r, tgt_quat_r], dim=-1))
    else:
        arm_ik.set_command(tgt_pos_r, ee_quat=ee_quat_r)  # ee_quat only for the display buffer
    return arm_ik.compute(ee_pos_r, ee_quat_r, jacobian, joint_pos)


def _grasp_report(robot, scene, ee_body_idx, finger_ids, box_half, grip_radius) -> str:
    """One-line grasp telemetry: ee<->box surface dist, gripper state, would grasp_success fire."""
    ee_w = robot.data.body_pos_w[:, ee_body_idx]          # (1,3) world
    box_w = scene["box"].data.root_pos_w[:, 0:3]          # (1,3) world
    surf = float(torch.norm(ee_w - box_w, dim=-1) - box_half)
    finger = float(robot.data.joint_pos[0, finger_ids[0]])
    closed = torch.tensor([finger < FINGER_CLOSED_THRESH], device=robot.device)
    half_t = torch.tensor([box_half], device=robot.device)
    fire = bool(grasp_success(ee_w, box_w, closed, half_t)[0])  # env's shipped radius
    gst = "CLOSED" if bool(closed[0]) else "open"
    near = "YES" if surf < grip_radius else "no "
    return (f"surface_dist={surf:+.3f}m  gripper={gst}  within {grip_radius:.2f}m? {near}  "
            f"grasp_success(@{GRIP_RADIUS_M:.2f})={'FIRE' if fire else '----'}")


# ── Scripted magnetic pickup (port of demo_pickup onto the light scene) ────────────────
DELIVER_RADIUS_M = 1.5   # box xy within this of the zone centre = delivered (matches env)


def _drive_toward(target_xy, base_xy, yaw, lin_gain: float = 1.0,
                  ang_gain: float = 2.0, face_tol: float = 0.6) -> tuple[float, float, float]:
    """Scripted base intent toward target_xy: (v_forward, yaw_rate, dist).

    Forward speed scaled by cos(heading_err) so the base progresses while turning; hard-stop forward
    only when nearly perpendicular. Mirrors the soft facing gate in scripts/demo_pickup._action.
    """
    import math
    dx, dy = target_xy[0] - base_xy[0], target_xy[1] - base_xy[1]
    dist = math.hypot(dx, dy)
    err = math.atan2(math.sin(math.atan2(dy, dx) - yaw), math.cos(math.atan2(dy, dx) - yaw))
    yaw_rate = max(-1.0, min(1.0, ang_gain * err))
    facing = max(0.0, math.cos(err))
    v_fwd = (max(0.0, min(1.0, lin_gain * dist)) * facing) if abs(err) < face_tol else 0.0
    return v_fwd, yaw_rate, dist


def _base_world_cmd(v_fwd: float, yaw_rate: float, yaw: float,
                    lin: float, ang: float, device) -> torch.Tensor:
    """Body-frame (forward, yaw) -> world prismatic_x/y + revolute_z velocity target (1,3)."""
    import math
    cy, sy = math.cos(yaw), math.sin(yaw)
    return torch.tensor([[v_fwd * lin * cy, v_fwd * lin * sy, yaw_rate * ang]], device=device)


def run_pickup() -> None:
    """Scripted magnetic pickup on the light scene: drive to box -> grab -> carry to zone -> deliver."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, render_interval=max(1, args_cli.render_every))
    sim = SimulationContext(sim_cfg)
    zone_xy = (float(args_cli.zone_x), float(args_cli.zone_y))
    sim.set_camera_view(eye=(-4.0, -4.0, 3.5), target=(args_cli.box_dist, -1.0, 0.3))
    scene_cfg = MiniSceneCfg(num_envs=1, env_spacing=8.0)
    size = float(args_cli.box_size)
    box_start = (args_cli.box_dist, 0.0, size / 2.0)
    scene_cfg.box = _item_cfg("box", size, BOX_MASS.get(size, 2.0), box_start)
    if not args_cli.no_rack:
        scene_cfg.rack_0 = _rack_cfg(0, (args_cli.box_dist + 0.6, 0.0, 0.0))
    scene_cfg.zone = AssetBaseCfg(   # visual-only delivery pad (no collision, like the env zones)
        prim_path="{ENV_REGEX_NS}/Zone",
        spawn=sim_utils.CuboidCfg(
            size=(2.0, 2.0, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.85, 0.25)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(zone_xy[0], zone_xy[1], 0.01)),
    )
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    box_half = size / 2.0

    robot = scene["robot"]
    box = scene["box"]
    ids0 = torch.tensor([0], device=sim.device)
    base_ids, _ = robot.find_joints(BASE_JOINTS, preserve_order=True)
    arm_ids, _ = robot.find_joints(ARM_JOINT_RE, preserve_order=True)
    finger_ids, _ = robot.find_joints(FINGER_JOINT_RE, preserve_order=True)
    base_link_idx = robot.body_names.index("base_link")
    ee_body_idx = robot.body_names.index(EE_BODY)
    arm_home = robot.data.default_joint_pos[:, arm_ids].clone()           # arm frozen here
    finger_closed = torch.full((1, len(finger_ids)), GRIP_CLOSE, device=sim.device)
    box_start_w = box.data.root_state_w[0:1].clone()                     # re-place box per episode

    print(f"[v2-pickup] box {size}m, zone={zone_xy}, grip_radius={GRIP_RADIUS_M:.2f}m. arm FROZEN, "
          f"gripper CLOSED, scripted base. {args_cli.episodes} episode(s).")
    if size < 0.52:
        print(f"[v2-pickup] WARNING: box {size}m may sit below the frozen hand's reach (~0.59 m up). "
              f"If it never grabs, retry with --box_size 0.52.")

    for ep in range(args_cli.episodes):
        if ep > 0:
            box.write_root_state_to_sim(box_start_w.clone(), env_ids=ids0)
        holding = False
        grabbed_at = delivered_at = -1
        for step in range(args_cli.max_steps):
            if not simulation_app.is_running():
                break
            base_p = robot.data.body_pos_w[0, base_link_idx]
            base_xy = (float(base_p[0]), float(base_p[1]))
            q = robot.data.body_quat_w[0, base_link_idx]
            yaw = float(torch.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                                    1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)))
            ee_w = robot.data.body_pos_w[:, ee_body_idx]        # (1,3) panda_hand world
            box_w = box.data.root_pos_w[:, 0:3]                 # (1,3) box world

            if not holding:   # magnetic grasp: SAME check the env uses (gripper held closed all run)
                near = grasp_success(ee_w, box_w, torch.tensor([True], device=sim.device),
                                     torch.tensor([box_half], device=sim.device))
                if bool(near[0]):
                    holding, grabbed_at = True, step
                    print(f"[v2-pickup] ep{ep} GRABBED at step {step}")

            target_xy = zone_xy if holding else (float(box_w[0, 0]), float(box_w[0, 1]))
            v_fwd, yaw_rate, dist = _drive_toward(target_xy, base_xy, yaw)
            robot.set_joint_velocity_target(
                _base_world_cmd(v_fwd, yaw_rate, yaw, args_cli.lin, args_cli.ang, sim.device),
                joint_ids=base_ids,
            )
            robot.set_joint_position_target(finger_closed, joint_ids=finger_ids)
            robot.write_joint_state_to_sim(arm_home, torch.zeros_like(arm_home), joint_ids=arm_ids)

            if holding:   # carry: weld box to the hand (teleport to panda_hand, zero vel) = env carry
                state = box.data.root_state_w[0:1].clone()
                state[:, 0:3] = ee_w
                state[:, 7:13] = 0.0
                box.write_root_state_to_sim(state, env_ids=ids0)

            scene.write_data_to_sim()
            sim.step()
            scene.update(dt=sim.get_physics_dt())

            if holding:
                bxy = box.data.root_pos_w[0, :2]
                d_zone = float(torch.hypot(bxy[0] - zone_xy[0], bxy[1] - zone_xy[1]))
                if d_zone < DELIVER_RADIUS_M:
                    delivered_at = step
                    print(f"[v2-pickup] ep{ep} DELIVERED at step {step} (box in zone, {d_zone:.2f}m)")
                    break
            if step % 40 == 0:
                print(f"[v2-pickup] ep{ep} step {step:4d} {'carry->zone' if holding else 'approach box'} "
                      f"dist={dist:.2f}m hold={int(holding)}")
        result = "DELIVERED" if delivered_at >= 0 else ("GRABBED-only" if grabbed_at >= 0 else "no-grab")
        print(f"[v2-pickup] ep{ep} result: {result} (grab@{grabbed_at}, deliver@{delivered_at})")


def _disable_arm_collisions(stage, robot_prim_path: str) -> None:
    """Turn OFF collision on every Franka arm/hand link. MUST run before sim.reset() so the OFF
    state bakes into the GPU PhysX cook — a runtime USD collisionEnabled toggle does NOT reliably
    propagate on the GPU pipeline. Mirrors env._disable_arm_collisions: the grasp is magnetic
    (proximity, not contact) and the carried box rides at the hand, so the arm colliders only catch
    the box/rack and fling the arm (the 'jitter banyak gerak random' while holding). Base/chassis
    colliders (no '/panda_') stay intact."""
    from pxr import Usd, UsdPhysics
    root = stage.GetPrimAtPath(robot_prim_path)
    for prim in Usd.PrimRange(root):
        if "/panda_" in prim.GetPath().pathString and prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)


def _set_box_collision(stage, box_prim_path: str, enabled: bool) -> None:
    """Toggle collision on the box prim. OFF while carried so the teleported box can't transmit
    contact force to the arm (smooth movement while holding) — mirrors env._set_box_collision."""
    from pxr import Usd, UsdPhysics
    root = stage.GetPrimAtPath(box_prim_path)
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(enabled)


def main() -> None:
    """Spawn the mini scene, then loop: keyboard -> base vel + arm IK + gripper, print grasp state."""
    sim, scene, box_half = _build_scene()
    # Disable the arm-link colliders BEFORE play so it bakes into the GPU cook (the only reliable
    # path). With them off the carried box transmits no contact -> no jitter while holding.
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    _disable_arm_collisions(stage, "/World/envs/env_0/Robot")
    sim.reset()
    robot = scene["robot"]
    grip_radius = args_cli.grip_radius if args_cli.grip_radius is not None else GRIP_RADIUS_M

    base_ids, _ = robot.find_joints(BASE_JOINTS, preserve_order=True)
    base_link_idx = robot.body_names.index("base_link")
    arm_ids, _ = robot.find_joints(ARM_JOINT_RE, preserve_order=True)
    finger_ids, _ = robot.find_joints(FINGER_JOINT_RE, preserve_order=True)
    ee_body_idx = robot.body_names.index(EE_BODY)
    # Robust Jacobian indexing (welded base may still report a floating Jacobian — see drive_robot.py).
    jac = robot.root_physx_view.get_jacobians()
    floating_jac = jac.shape[-1] == robot.num_joints + 6
    jac_joint_ids = [i + 6 for i in arm_ids] if floating_jac else list(arm_ids)
    ee_jacobi_idx = ee_body_idx if floating_jac else ee_body_idx - 1
    # 'down' = pose IK (top-down) with dls damping (--ik_damp, default 0.05 vs the 0.01 default) so the
    # arm resolves smoothly near the wrist-limit singularity instead of flipping ("langsung jatuh"). It
    # under-tracks orientation a little (hand tilts) rather than collapsing. Too-high damping suppresses
    # xy near the extended config ("cuma bisa z") -> tune --ik_damp. 'free' = position-only (orientation
    # unconstrained — hand twists 'di dalam'), the no-collapse fallback, always lambda 0.01.
    pose_mode = args_cli.orient == "down"
    arm_ik = DifferentialIKController(
        DifferentialIKControllerCfg(
            command_type="pose" if pose_mode else "position",
            use_relative_mode=False, ik_method="dls",
            ik_params={"lambda_val": float(args_cli.ik_damp) if pose_mode else 0.01},
        ),
        num_envs=1, device=sim.device,
    )
    arm_ik.reset()

    base_kb = Se2Keyboard(Se2KeyboardCfg(
        v_x_sensitivity=args_cli.lin, v_y_sensitivity=args_cli.strafe,
        omega_z_sensitivity=args_cli.ang, sim_device=sim.device,
    ))
    arm_kb = Se3Keyboard(Se3KeyboardCfg(
        pos_sensitivity=args_cli.ee_sens, rot_sensitivity=0.0,
        gripper_term=True, sim_device=sim.device,
    ))
    print(f"[v2] LIGHT teleop: 1 box ({args_cli.box_size}m){'' if args_cli.no_rack else ' + 1 rack'}, "
          f"NO camera, render every {args_cli.render_every} steps. grasp radius={grip_radius:.2f}m.")
    print("[v2] Focus the viewport. BASE: arrows + Z/X. ARM: W/S A/D Q/E, K=gripper. Ctrl-C to quit.")

    # Settle to the tucked spawn pose, then capture the absolute EE hold target (keys accumulate onto it).
    for _ in range(30):
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt=sim.get_physics_dt())
    # Capture the EE hold target RELATIVE TO base_link (the moving chassis), NOT the welded root.
    # Root is pinned at world origin (#1268), so a root-relative target stays world-fixed and the
    # arm drags back to origin as the base drives away. base-relative = the arm rides the chassis.
    base_pos_w = robot.data.body_pos_w[:, base_link_idx]
    base_quat_w = robot.data.body_quat_w[:, base_link_idx]
    ee_pos_w = robot.data.body_pos_w[:, ee_body_idx]
    ee_quat_w = robot.data.body_quat_w[:, ee_body_idx]
    ee_target, ee_quat_des = subtract_frame_transforms(base_pos_w, base_quat_w, ee_pos_w, ee_quat_w)
    ee_target, ee_quat_des = ee_target.clone(), ee_quat_des.clone()
    calib = bool(args_cli.calib)
    # EE target clamp: a box around the spawn pose, ASYMMETRIC (Z tighter than XY). In --calib the
    # box is opened wide so you can freely sweep the reachable envelope for fitting.
    rxy, rz = (1.5, 1.5) if calib else (args_cli.reach_xy, args_cli.reach_z)
    reach = torch.tensor([rxy, rxy, rz], device=sim.device)
    ee_min, ee_max = ee_target - reach, ee_target + reach
    # Per-joint limits (rad) from the Franka USD. HARD limits feed the calibration margin metric;
    # SOFT limits (joint_margin fraction of the range, centred) are what we clamp the IK output to,
    # so the arm stops short of the hard stops (no chatter). The controller does NOT clamp its own.
    jlim = robot.data.joint_pos_limits[:, arm_ids]   # (1, 7, 2) [min, max]
    hard_min, hard_max = jlim[..., 0], jlim[..., 1]
    j_mid, j_half = 0.5 * (hard_min + hard_max), 0.5 * (hard_max - hard_min)
    margin = float(min(1.0, max(0.05, args_cli.joint_margin)))
    jmin, jmax = j_mid - j_half * margin, j_mid + j_half * margin
    # Shoulder centre for the radial workspace clamp = panda_link0 in base_link frame (constant;
    # arm rigidly mounted on the chassis). Fitted r_min/r_max are measured from here.
    sh_idx = robot.body_names.index("panda_link0")
    center_b, _ = subtract_frame_transforms(
        base_pos_w, base_quat_w, robot.data.body_pos_w[:, sh_idx], robot.data.body_quat_w[:, sh_idx]
    )
    r_max, r_min = float(args_cli.reach_r), float(args_cli.reach_rmin)  # r_max<=0 -> radial OFF
    if calib:
        r_max = 0.0   # calibration: free the envelope (no radial clamp) so the full reach is swept
    calib_f = calib_w = None
    if calib:
        cpath = PROJECT_ROOT / args_cli.calib
        cpath.parent.mkdir(parents=True, exist_ok=True)
        calib_f = open(cpath, "w", newline="")
        calib_w = csv.writer(calib_f)
        calib_w.writerow(["step", "ee_x_b", "ee_y_b", "ee_z_b", "radius_m",
                          "min_joint_margin", "pose_err_m", "manipulability", "gripper"])
        print(f"[v2][CALIB] envelope FREE (clamps off). logging -> {cpath}")
        print(f"[v2][CALIB] shoulder center (base frame) = {[round(v, 3) for v in center_b[0].tolist()]}")
    # Smoothed joint-target state (EXP low-pass): start AT the settled arm pose so the first
    # command doesn't jump. beta in (0,1]; lower = calmer. Kills high-freq IK twitch / flail.
    beta = float(min(1.0, max(0.01, args_cli.arm_smooth)))
    smooth_targets = robot.data.joint_pos[:, arm_ids].clone()
    arm_freeze_pose = smooth_targets.clone()   # snapshot held during --freeze_on_grip (set at grab)
    # Auto-pick state (active arm + magnetic grasp, like the env): when the (moving) hand gets within
    # grip_radius of the box surface, grab + carry the box welded to the hand (proximity-only).
    box = scene["box"]
    ids0 = torch.tensor([0], device=sim.device)
    holding = False
    prev_want_open = True          # Se3Keyboard gripper default = OPEN; track for the K press EDGE
    release_latch = False          # blocks re-grab after a release until the hand leaves the box
    box_path = box.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")

    step = 0
    while simulation_app.is_running():
        cmd = base_kb.advance().unsqueeze(0)             # (1,3) [vx,vy,wz] body-frame intent
        q = robot.data.body_quat_w[0, base_link_idx]
        yaw = torch.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2))
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        vx_b, vy_b = cmd[0, 0].clone(), cmd[0, 1].clone()
        cmd[0, 0] = vx_b * cy - vy_b * sy                # project body fwd -> world-x (prismatic_x)
        cmd[0, 1] = vx_b * sy + vy_b * cy                # -> world-y (prismatic_y)
        robot.set_joint_velocity_target(cmd, joint_ids=base_ids)

        arm_cmd = arm_kb.advance()                       # (7,) [dx,dy,dz, rx,ry,rz, grip]
        # Current EE (base frame) — for the leash, freeze re-anchor + telemetry. Computed every step.
        ee_now_b, _ = subtract_frame_transforms(
            robot.data.body_pos_w[:, base_link_idx], robot.data.body_quat_w[:, base_link_idx],
            robot.data.body_pos_w[:, ee_body_idx], robot.data.body_quat_w[:, ee_body_idx])
        if holding and args_cli.freeze_on_grip:
            # HARD freeze: arm LOCKED at the grab pose, arm keys ignored — only the base drives (carry
            # to the zone without the arm wandering into a rack). ee_target tracks the live EE so the
            # release (K) resumes IK without a jump.
            ee_target = ee_now_b.clone()
            robot.set_joint_position_target(arm_freeze_pose, joint_ids=arm_ids)
        else:
            ee_target += arm_cmd[:3].unsqueeze(0)
            # Workspace-box guard on the EE target (base frame): clamp backward x (S can't drive the
            # gripper into the base/rack behind — "mundur nabrak"; +x forward unbounded) and cap z
            # up/down (Q not "tinggi-tinggi banget", E not below the floor). Past a bound the extra
            # key input is silently dropped — the leash keeps the clamped target near the live EE.
            ee_target[..., 0] = ee_target[..., 0].clamp_min(args_cli.reach_x_back)
            ee_target[..., 2] = ee_target[..., 2].clamp(args_cli.reach_z_min, args_cli.reach_z_max)
            # Lead/leash (NON-locking): bound the commanded target to within `lead` m of the LIVE EE.
            # ee_target is a free accumulator — without this, z-down past the floor buries the target
            # unreachably and the arm folds and can't climb back ("jatuh, gak bisa berdiri"). Clamping
            # to the LIVE EE (not a snapshot) means the reverse key instantly shrinks the lead and
            # recovers. No-op in the dexterous zone (n<=lead); engages only at the reach boundary.
            if args_cli.lead > 0.0:
                lead_vec = ee_target - ee_now_b
                n = torch.norm(lead_vec, dim=-1, keepdim=True)
                ee_target = ee_now_b + lead_vec * (n.clamp(max=args_cli.lead) / n.clamp_min(1e-6))
            # Radial workspace clamp (couples x/y/z), OFF by default (--reach_r 0): the 0.95 shell sat
            # in the near-singular zone and jittered. Re-enable with --reach_r 0.90 for an outer guard.
            if r_max > 0.0:
                v = ee_target - center_b
                r = torch.norm(v, dim=-1, keepdim=True).clamp_min(1e-6)
                ee_target = center_b + v * (r.clamp(r_min, r_max) / r)
            arm_targets = _arm_ik_targets(
                robot, arm_ik, arm_ids, jac_joint_ids, ee_body_idx, ee_jacobi_idx,
                base_link_idx, ee_target, ee_quat_des
            )
            # Clamp the IK output to the hard joint limits BEFORE smoothing — the controller does not,
            # so an unreachable target (esp. vertical Z) would push joint4/joint6 past range + oscillate.
            arm_targets = torch.clamp(arm_targets, jmin, jmax)
            # EXP low-pass the IK joint targets — smooths twitch from fast key slew + base transients.
            smooth_targets = beta * arm_targets + (1.0 - beta) * smooth_targets
            # Hard per-step delta clamp referenced to the ACTUAL joint pos: commanded target can never
            # be > max_joint_step rad from where the arm IS, so a near-singular IK flip can't fling it
            # in one step ("langsung jatuh"). Decouples flip-safety from --ik_damp.
            q_cur = robot.data.joint_pos[:, arm_ids]
            smooth_targets = q_cur + torch.clamp(
                smooth_targets - q_cur, -args_cli.max_joint_step, args_cli.max_joint_step)
            robot.set_joint_position_target(smooth_targets, joint_ids=arm_ids)
        # ── Magnetic grab: fires on PROXIMITY alone (no key timing). Se3Keyboard's gripper is a TOGGLE
        # whose default level is OPEN, so we trigger RELEASE on its press EDGE (a K tap), not its level —
        # otherwise the default-open level spuriously releases every step and breaks the freeze (arm
        # drifts up + random, box flies up). Re-grab is blocked until the hand leaves the box again.
        ee_w = robot.data.body_pos_w[:, ee_body_idx]
        box_w = box.data.root_pos_w[:, 0:3]
        surf = float(torch.norm(ee_w - box_w, dim=-1)[0]) - box_half   # ee<->box SURFACE dist
        want_open = float(arm_cmd[-1]) > 0.0
        k_edge = want_open != prev_want_open                # True only on the frame K is tapped
        prev_want_open = want_open
        if not holding and surf < grip_radius and not release_latch:
            holding = True
            _set_box_collision(stage, box_path, False)   # OFF: carried box can't push the arm
            ee_target = ee_now_b.clone()                 # soft freeze: re-anchor target at the grab pose
            smooth_targets = robot.data.joint_pos[:, arm_ids].clone()  # -> arm holds dead-still
            arm_freeze_pose = smooth_targets.clone()     # snapshot for --freeze_on_grip HARD lock
            print(f"[v2] GRABBED at step {step}, surface_dist={surf:+.3f}m")
        elif holding and k_edge:                         # K tap -> release
            holding = False
            release_latch = True                         # block re-grab until the hand clears the box
            _set_box_collision(stage, box_path, True)    # back ON: box collides + falls again
            print(f"[v2] RELEASED at step {step}")
        if surf >= grip_radius:
            release_latch = False                        # re-arm grab once the hand is clear of the box
        # Gripper visual: CLOSED while holding (magnet), OPEN otherwise.
        finger = GRIP_CLOSE if holding else GRIP_OPEN
        robot.set_joint_position_target(
            torch.full((1, len(finger_ids)), finger, device=sim.device), joint_ids=finger_ids)
        if holding:   # carry: float box just in FRONT of the hand (along base yaw), NOT welded onto
            # it. Arm colliders are off (baked at startup) so it transmits no contact either way, but
            # floating-in-front keeps the box from visually penetrating the hand and sitting too high.
            # zero vel so it can't accumulate momentum.
            cy, sy = torch.cos(yaw), torch.sin(yaw)
            state = box.data.root_state_w[0:1].clone()
            state[:, 0] = ee_w[:, 0] + cy * (box_half + CARRY_GAP_M)
            state[:, 1] = ee_w[:, 1] + sy * (box_half + CARRY_GAP_M)
            state[:, 2] = ee_w[:, 2]
            state[:, 7:13] = 0.0
            box.write_root_state_to_sim(state, env_ids=ids0)

        scene.write_data_to_sim()
        sim.step()
        scene.update(dt=sim.get_physics_dt())

        if step % 40 == 0:  # ~5 Hz: target-vs-actual EE (base frame) + grasp telemetry
            t, a = ee_target[0].tolist(), ee_now_b[0].tolist()
            print(f"[v2] tgt(b)=[{t[0]:+.3f} {t[1]:+.3f} {t[2]:+.3f}]  "
                  f"ee(b)=[{a[0]:+.3f} {a[1]:+.3f} {a[2]:+.3f}]  "
                  f"err(t-ee)=[{t[0]-a[0]:+.3f} {t[1]-a[1]:+.3f} {t[2]-a[2]:+.3f}]  "
                  f"|err|={((t[0]-a[0])**2+(t[1]-a[1])**2+(t[2]-a[2])**2)**0.5:.3f}m")
            print(f"[v2] {_grasp_report(robot, scene, ee_body_idx, finger_ids, box_half, grip_radius)}")
        if calib_w is not None:
            # Singularity sensor: manipulability = sqrt(det(Jp Jpᵀ)) on the 3x7 position Jacobian.
            J = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, jac_joint_ids]  # (1,6,7)
            Jp = J[:, 0:3, :]
            manip = float(torch.sqrt(torch.det(Jp @ Jp.transpose(1, 2)).clamp_min(0.0))[0])
            q_now = robot.data.joint_pos[:, arm_ids]
            mjm = float((torch.minimum(q_now - hard_min, hard_max - q_now) / j_half).min())
            r_now = float(torch.norm(ee_now_b - center_b, dim=-1)[0])
            perr = float(torch.norm(ee_target - ee_now_b, dim=-1)[0])
            calib_w.writerow([step, *[round(float(x), 4) for x in ee_now_b[0].tolist()],
                              round(r_now, 4), round(mjm, 4), round(perr, 4), round(manip, 6),
                              round(float(robot.data.joint_pos[0, finger_ids[0]]), 4)])
            if step % 50 == 0:
                calib_f.flush()
        step += 1

    if calib_f is not None:
        calib_f.close()
        print("[v2][CALIB] log closed.")


if __name__ == "__main__":
    run_pickup() if args_cli.pickup else main()
    simulation_app.close()
