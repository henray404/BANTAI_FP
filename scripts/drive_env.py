# drive_env.py — Manual keyboard teleop through the REAL RL env (WarehouseGymEnv).
#
# Unlike scripts/drive_robot.py (which hand-rolls a DifferentialIKController on a bare
# InteractiveScene), this drives the actual env action term: action (6,) =
# [base_lin, base_ang, ee_dx, ee_dy, ee_dz, gripper]. Isaac Lab's DifferentialInverseKinematics
# action term handles ALL the frame/Jacobian/base-offset bookkeeping internally — so the EE
# tracks commands correctly on the welded mobile base (no hand-rolled frame bugs). This also
# exercises the exact wiring P2/P3 consume.
#
# The env uses the onboard camera, so --enable_cameras is forced on (pin NVIDIA driver 580.88;
# see bugs_errors/2026-05-22_sdp-camera-crash-blackwell.md). AppLauncher created first.
#
# Usage:
#   conda activate isaaclab
#   python scripts/drive_env.py                 # windowed (focus viewport to send keys)
#   python scripts/drive_env.py --ee_sens 0.08  # faster EE
#
# Keys (window must be focused):
#   BASE:  Arrow Up/Down = forward/back   Z/X = yaw +/-
#   ARM :  W/S = EE +x/-x   A/D = EE +y/-y   Q/E = EE +z/-z   K = toggle gripper
#   L   :  reset arm command to zero

"""Keyboard teleop through WarehouseGymEnv — drives the real (6,) action / IK term."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

# ── CLI + AppLauncher ─────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Keyboard teleop through the warehouse RL env")
parser.add_argument("--lin", type=float, default=1.0, help="base forward sensitivity ([-1,1] action)")
parser.add_argument("--ang", type=float, default=1.0, help="base yaw sensitivity ([-1,1] action)")
parser.add_argument("--ee_sens", type=float, default=0.05,
                    help="EE action per held arm key. Env IK treats it as a metre delta per 10Hz "
                         "control step; 0.05 ~= 0.5 m/s.")
parser.add_argument("--cam", type=int, default=0,
                    help="Save the onboard camera frame to _cam_debug/ every N control steps "
                         "(0 = off). Use to SEE what the robot's camera captures.")
parser.add_argument("--chase", type=int, default=1,
                    help="3rd-person chase camera that follows the robot like a game (1=on, 0=off). "
                         "Moves the VIEWPORT camera only — does NOT touch the onboard 'pixels' obs.")
parser.add_argument("--chase_back", type=float, default=6.0,
                    help="Chase camera distance behind the robot (metres).")
parser.add_argument("--chase_height", type=float, default=4.0,
                    help="Chase camera height above the robot (metres).")
parser.add_argument("--log", type=str, default="",
                    help="Write a per-step diagnostic CSV to this path (relative to project root). "
                         "Empty = off. Captures action, ee_pos, base roll/pitch/yaw, gripper, holding.")
parser.add_argument("--debug_reward", action="store_true",
                    help="Print the per-step reward breakdown (which term drives the step reward) ~1/s. "
                         "Use to SEE why return is what it is — does grasp fire, is approach shrinking.")
parser.add_argument("--record", type=str, default="",
                    help="Write a FULL replayable run (all joints + poses + box + metadata) to this "
                         "path stem. Produces <path>.csv + <path>.meta.json playable via replay_csv.py. "
                         "Records one episode (until done), then stops. Empty = off.")
parser.add_argument("--seed", type=int, default=None,
                    help="Env seed (stored in the recorded metadata so replay reproduces the scene).")
AppLauncher.add_app_launcher_args(parser)
# Teleop sensitivities are tunable in configs/teleop.yaml — load them as argparse defaults so a
# CLI flag still wins, the YAML overrides the baked-in fallbacks, and editing the file needs no
# code change. Missing file/keys keep the hardcoded defaults above.
_TELEOP_CFG = Path(__file__).resolve().parents[1] / "configs" / "teleop.yaml"
if _TELEOP_CFG.exists():
    try:
        import yaml
        _tele = yaml.safe_load(_TELEOP_CFG.read_text(encoding="utf-8")) or {}
    except ImportError:
        import ruamel.yaml as _ryaml
        _tele = _ryaml.YAML(typ="safe").load(_TELEOP_CFG.read_text(encoding="utf-8")) or {}
    _tele = _tele.get("teleop", {}) or {}
    parser.set_defaults(**{k: _tele[k] for k in
                           ("ee_sens", "lin", "ang", "chase", "chase_back", "chase_height")
                           if k in _tele})
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # warehouse env always builds the onboard camera

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Disable the PhysX Support UI extension: its selection manipulator throws
# "Accessed invalid null prim" every frame when env resets re-write box/robot
# prims, spamming tracebacks to the console and stuttering the render loop.
# It is a GUI convenience only — the env does not need it.
import omni.kit.app  # noqa: E402
omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
    "omni.physx.supportui", False
)

# ── Imports (after AppLauncher) ───────────────────────────────────────
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg, Se3Keyboard, Se3KeyboardCfg  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from env.warehouse_env import WarehouseEnvCfg, WarehouseGymEnv  # noqa: E402
from recording.recorder import TrajectoryRecorder  # noqa: E402
from recording.state_extractor import build_metadata, step_row  # noqa: E402


def _save_frame(frame: np.ndarray, path: Path) -> None:
    """Save an (H,W,3) uint8 frame to PNG (imageio -> PIL -> .npy fallback)."""
    try:
        import imageio.v3 as iio
        iio.imwrite(path, frame)
    except Exception:
        try:
            from PIL import Image
            Image.fromarray(frame).save(path)
        except Exception:
            np.save(path.with_suffix(".npy"), frame)  # always works; view later


def _build_action(base_cmd: torch.Tensor, arm_cmd: torch.Tensor, num_envs: int) -> np.ndarray:
    """Assemble the (num_envs, 6) env action from SE(2) base + SE(3) arm keyboard commands.

    base_cmd = [vx, vy, wz]; arm_cmd = [dx, dy, dz, rx, ry, rz, grip].
    action   = [base_lin(=vx), base_ang(=wz), ee_dx, ee_dy, ee_dz, gripper(=grip)].
    """
    action = np.zeros((num_envs, 6), dtype=np.float32)
    action[:, 0] = float(base_cmd[0])      # base linear (forward)
    action[:, 1] = float(base_cmd[2])      # base angular (yaw)
    action[:, 2:5] = arm_cmd[:3].cpu().numpy()   # EE position delta (base frame)
    action[:, 5] = float(arm_cmd[-1])      # gripper: >0 open, <=0 close
    return action


def _base_euler_deg(env) -> tuple[float, float, float]:
    """Return base_link (roll, pitch, yaw) in degrees from its world quaternion (moving chassis)."""
    q = env._env.scene["robot"].data.body_quat_w[0, env._base_link_idx]  # (w, x, y, z)
    w, x, y, z = (float(v) for v in q)
    roll = math.degrees(math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))))
    yaw = math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    return roll, pitch, yaw


def _chase_cam(env, obs: dict, back: float, height: float) -> None:
    """Point the viewport camera behind+above the robot, looking at it (3rd-person follow).

    Reads the MOVING base_link world pose (NOT root_pos_w — fixed-root robot, IsaacLab #1268)
    and the heading to place the camera behind the robot's facing direction.
    """
    base_w = env._env.scene["robot"].data.body_pos_w[0, env._base_link_idx]  # world xyz (moving)
    bx, by, bz = float(base_w[0]), float(base_w[1]), float(base_w[2])
    cos_y, sin_y = float(obs["heading"][0][0]), float(obs["heading"][0][1])  # [cos(yaw), sin(yaw)]
    eye = (bx - back * cos_y, by - back * sin_y, bz + height)
    target = (bx + cos_y, by + sin_y, bz + 0.5)
    env._env.sim.set_camera_view(eye=eye, target=target)


def main() -> None:
    """Build the env, then loop: keyboard -> (6,) action -> env.step; print EE / grasp state."""
    cfg = WarehouseEnvCfg()
    cfg.scene.num_envs = 1
    env = WarehouseGymEnv(cfg=cfg, arm_active=True)   # teleop: force active arm regardless of stage
    obs, _ = env.reset(seed=args_cli.seed)
    print(f"[drive_env] reset ok. obs keys = {sorted(obs.keys())}")
    print(f"[drive_env] action = [base_lin, base_ang, ee_dx, ee_dy, ee_dz, gripper], shape (6,)")

    base_kb = Se2Keyboard(
        Se2KeyboardCfg(v_x_sensitivity=args_cli.lin, v_y_sensitivity=0.0,
                       omega_z_sensitivity=args_cli.ang, sim_device=env.device)
    )
    arm_kb = Se3Keyboard(
        Se3KeyboardCfg(pos_sensitivity=args_cli.ee_sens, rot_sensitivity=0.0,
                       gripper_term=True, sim_device=env.device)
    )
    print(base_kb)
    print("[drive_env] Focus the viewport. BASE: arrows + Z/X. ARM: W/S A/D Q/E, K=gripper. Ctrl-C to quit.")

    cam_dir = PROJECT_ROOT / "_cam_debug"
    if args_cli.cam > 0:
        cam_dir.mkdir(exist_ok=True)
        print(f"[drive_env] saving onboard camera every {args_cli.cam} steps -> {cam_dir}")

    log_f = None
    log_w = None
    if args_cli.log:
        log_path = PROJECT_ROOT / args_cli.log
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(log_path, "w", newline="")
        log_w = csv.writer(log_f)
        log_w.writerow(["t", "step", "a_base_lin", "a_base_ang", "a_ee_dx", "a_ee_dy", "a_ee_dz",
                        "a_grip", "ee_x", "ee_y", "ee_z", "base_roll_deg", "base_pitch_deg",
                        "base_yaw_deg", "base_wx", "base_wy", "gripper", "holding"])
        print(f"[drive_env] logging per-step diagnostic CSV -> {log_path}")

    # Full replayable recorder (one episode). Separate from --log: same playable format as
    # scripts/record_scenario.py, so a teleop run can be replayed via scripts/replay_csv.py.
    rec = None
    if args_cli.record:
        meta = build_metadata(env, seed=args_cli.seed, policy="teleop",
                              run_id=Path(args_cli.record).name)
        rec = TrajectoryRecorder(args_cli.record, metadata=meta)
        print(f"[drive_env] recording FULL replayable run -> {rec.csv_path} (one episode)")

    t0 = time.time()
    step = 0
    while simulation_app.is_running():
        action = _build_action(base_kb.advance(), arm_kb.advance(), env.num_envs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated[0]) or bool(truncated[0])

        # Record the full per-step state BEFORE any reset (so the run is one clean episode).
        if rec is not None:
            rec.add(step_row(env, step=step, t=time.time() - t0, action=action, reward=reward,
                             terminated=terminated, truncated=truncated, info=info))
            if done:
                rec.set_summary({"steps": step + 1, "success": int(bool(terminated[0]))})
                rec.close()
                print(f"[drive_env] recorded run saved -> {rec.csv_path} ({step + 1} steps). "
                      f"Replay: python scripts/replay_csv.py --run {args_cli.record}")
                rec = None

        if done:
            obs, _ = env.reset()

        if args_cli.chase > 0:
            _chase_cam(env, obs, args_cli.chase_back, args_cli.chase_height)

        if args_cli.cam > 0 and step % args_cli.cam == 0:
            _save_frame(env.render(), cam_dir / f"cam_{step:05d}.png")

        if log_w is not None:
            roll, pitch, yaw = _base_euler_deg(env)
            base_w = env._env.scene["robot"].data.body_pos_w[0, env._base_link_idx]  # world xyz
            ee = obs["ee_pos"][0].tolist()
            a = [float(v) for v in action[0].tolist()]
            log_w.writerow([round(time.time() - t0, 3), step,
                            *[round(v, 4) for v in a],
                            round(ee[0], 4), round(ee[1], 4), round(ee[2], 4),
                            round(roll, 2), round(pitch, 2), round(yaw, 2),
                            round(float(base_w[0]), 4), round(float(base_w[1]), 4),
                            round(float(obs["gripper"][0]), 3), int(obs["holding"][0])])
            if step % 10 == 0:
                log_f.flush()

        if step % 10 == 0:  # env runs at 10Hz control -> ~1 Hz print
            ee = [round(v, 3) for v in obs["ee_pos"][0].tolist()]
            box = [round(v, 3) for v in obs["box_pos"][0].tolist()]
            grip = round(float(obs["gripper"][0]), 3)
            hold = int(obs["holding"][0])
            act = [round(float(v), 3) for v in action[0].tolist()]
            if any(abs(v) > 1e-6 for v in act) or hold:
                print(f"[drive_env] action={act}  ee_pos={ee} gripper={grip} holding={hold} box={box}")
            if args_cli.debug_reward:
                from env.reward_debug import reward_breakdown, format_breakdown
                print("           " + format_breakdown(reward_breakdown(env._env)))
        step += 1

    if log_f is not None:
        log_f.close()
    if rec is not None:  # quit before the episode ended — save what we have
        rec.set_summary({"steps": step, "success": 0, "note": "incomplete (quit before done)"})
        rec.close()
        print(f"[drive_env] recorded partial run -> {rec.csv_path} ({step} steps)")


if __name__ == "__main__":
    main()
    simulation_app.close()
