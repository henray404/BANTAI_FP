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
import sys
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
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # warehouse env always builds the onboard camera

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Imports (after AppLauncher) ───────────────────────────────────────
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg, Se3Keyboard, Se3KeyboardCfg  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from env.warehouse_env import WarehouseEnvCfg, WarehouseGymEnv  # noqa: E402


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


def main() -> None:
    """Build the env, then loop: keyboard -> (6,) action -> env.step; print EE / grasp state."""
    cfg = WarehouseEnvCfg()
    cfg.scene.num_envs = 1
    env = WarehouseGymEnv(cfg=cfg)
    obs, _ = env.reset()
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

    step = 0
    while simulation_app.is_running():
        action = _build_action(base_kb.advance(), arm_kb.advance(), env.num_envs)
        obs, reward, terminated, truncated, _ = env.step(action)
        if bool(terminated[0]) or bool(truncated[0]):
            obs, _ = env.reset()

        if args_cli.cam > 0 and step % args_cli.cam == 0:
            _save_frame(env.render(), cam_dir / f"cam_{step:05d}.png")

        if step % 10 == 0:  # env runs at 10Hz control -> ~1 Hz print
            ee = [round(v, 3) for v in obs["ee_pos"][0].tolist()]
            box = [round(v, 3) for v in obs["box_pos"][0].tolist()]
            grip = round(float(obs["gripper"][0]), 3)
            hold = int(obs["holding"][0])
            act = [round(float(v), 3) for v in action[0].tolist()]
            if any(abs(v) > 1e-6 for v in act) or hold:
                print(f"[drive_env] action={act}  ee_pos={ee} gripper={grip} holding={hold} box={box}")
        step += 1


if __name__ == "__main__":
    main()
    simulation_app.close()
