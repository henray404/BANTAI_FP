# demo_pickup.py — scripted full pickup→carry→sort demo (no training, no policy).
#
# Shows the magnetic-pickup behaviour end to end so you can SEE it work without training a policy:
#   drive base toward the commanded box  →  stop in front (arm is FROZEN, never reaches/knocks)  →
#   box is grabbed on proximity (invisible grip sized to the box)  →  carry to the colour-coded
#   zone (goal_id: fragile→orange, regular→cyan, heavy→purple)  →  env fires the delivered success.
#
# This is a hand-written base controller, NOT the RL policy — the real approach skill is learned in
# scripts/train_p3.py. It only steers [base_lin, base_ang]; the EE action is ignored (arm frozen by
# WarehouseGymEnv.step) and the gripper is held CLOSED so the magnetic grasp latches on proximity.
#
# AppLauncher MUST be created here, before any isaaclab imports from env/.
# See bugs_errors/2026-05-15_double-applaunch-crash.md.
#
# Usage:
#   conda activate isaaclab
#   python scripts/demo_pickup.py                  # Stage-2 (spawn near box): short clean approach
#   python scripts/demo_pickup.py --stage 3        # full chain (spawn north, long nav — may hit racks)
#   python scripts/demo_pickup.py --episodes 5

"""Scripted demo of the magnetic pickup → carry → colour-sort cycle."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import yaml

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Scripted pickup demo")
parser.add_argument("--stage", type=int, default=2, help="curriculum stage to demo (2 = spawn near box)")
parser.add_argument("--episodes", type=int, default=3, help="number of pickup episodes to run")
parser.add_argument("--max_steps", type=int, default=600, help="step cap per episode")
parser.add_argument("--log_every", type=int, default=5,
                    help="print a localization line every N control steps (10 Hz → 5 = every 0.5 s; 0 = off)")
parser.add_argument("--log_csv", type=str, default=None,
                    help="optional path to write the per-step localization log as CSV")
parser.add_argument("--config", type=str, default=None,
                    help="optional tuning YAML (see configs/demo_tuning.yaml) overriding controller/avoidance/grasp/spawn knobs")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# The env mounts a TiledCamera, which requires the rendering pipeline — force cameras on.
# Set on args_cli (add_app_launcher_args already registered --enable_cameras); passing it as a
# kwarg too makes AppLauncher raise "both provided common attributes: {'enable_cameras'}".
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from env.warehouse_env import WarehouseEnvCfg, WarehouseGymEnv  # noqa: E402
from env.layout_grid import RACK_POSITIONS, avoidance_heading  # noqa: E402

DELIVER_RADIUS = 1.4   # consider the box delivered once the base is within this of the zone centre
FACE_TOL = 0.6         # only drive forward when the heading error is below this (rad)

# Obstacle avoidance (potential field): repel the base from each of the 18 RACKS (not the 9 island
# centres — a 2-rack island is too coarse as one point, and a wide influence closes the 6 m aisles).
RACKS = [(rx, ry) for rx, ry, _ in RACK_POSITIONS]   # env-local rack xy
AVOID_INFLUENCE = 1.4  # rack repulsion reach (m): rack-half ~0.6 + robot-half ~0.5 + margin; keeps aisles open
AVOID_PUSH = 2.0       # repulsion gain (summed with the unit attraction vector) — firm enough to deflect
SKIP_NEAR_TARGET = 0.8  # ignore ONLY the rack the target box sits on (sibling rack 1.5 m away still repels)
ANG_GAIN = 2.0         # yaw P-gain: base_ang = clip(ANG_GAIN * heading_err)
LIN_GAIN = 1.0         # distance P-gain: base_lin = clip(LIN_GAIN * dist)

# Wedge recovery: if the base commands forward but barely moves for a few steps it is jammed on a
# rack — back out + turn to break contact, then resume normal steering. Prevents permanent freeze.
MOVE_EPS = 0.01        # m/step below this counts as "not moving"
RECOVER_AFTER = 6      # consecutive wedged steps (≈0.6 s) before triggering recovery
RECOVER_FOR = 12       # steps (≈1.2 s) of reverse+turn to unstick
RECOVER_LIN = -0.6     # reverse speed during recovery (base_lin)
RECOVER_ANG = 0.8      # turn rate during recovery (base_ang) — rotate while backing out

# Post-grasp back-out: the box sits ON a rack, so right after grabbing the base is buried in that
# rack. Reverse STRAIGHT out (retracing the approach into the open aisle) before navigating to the
# zone, else the strong rack repulsion just spins the pinned base in place.
BACKOUT_STEPS = 18     # steps (≈1.8 s) of straight reverse after grasping
BACKOUT_LIN = -0.9     # reverse speed during back-out (base_lin)


def _f(t) -> np.ndarray:
    """First-env row of an obs tensor as a numpy float array."""
    return t[0].detach().cpu().numpy().astype(float)


def _yaw_err(target_xy, base_xy, heading) -> tuple[float, float]:
    """Return (distance, signed heading error) from base toward target_xy."""
    dx, dy = target_xy[0] - base_xy[0], target_xy[1] - base_xy[1]
    dist = math.hypot(dx, dy)
    desired = math.atan2(dy, dx)
    cur = math.atan2(heading[1], heading[0])           # heading = [cos(yaw), sin(yaw)]
    err = math.atan2(math.sin(desired - cur), math.cos(desired - cur))
    return dist, err


def _action(target_xy, base_xy, heading) -> np.ndarray:
    """Base controller: steer toward (target attraction − island repulsion), drive when facing it.

    Gripper held CLOSED (gripper<0) so the magnetic grasp latches on proximity.
    """
    desired, dist = avoidance_heading(
        target_xy, base_xy, RACKS, AVOID_INFLUENCE, AVOID_PUSH, SKIP_NEAR_TARGET
    )
    cur = math.atan2(heading[1], heading[0])                # heading = [cos(yaw), sin(yaw)]
    err = math.atan2(math.sin(desired - cur), math.cos(desired - cur))
    ang = float(np.clip(ANG_GAIN * err, -1.0, 1.0))
    # Soft facing gate: drive forward scaled by cos(err) so the base makes progress WHILE turning
    # (a hard "lin=0 unless facing" gate makes it spin in place forever when avoidance keeps the
    # heading error above the threshold). Only hard-stop when nearly perpendicular (|err| > FACE_TOL).
    facing = max(0.0, math.cos(err))
    lin = float(np.clip(LIN_GAIN * dist, 0.0, 1.0)) * facing if abs(err) < FACE_TOL else 0.0
    # [base_lin, base_ang, ee_dx, ee_dy, ee_dz, gripper] — ee ignored (arm frozen), gripper<0 = closed.
    return np.array([lin, ang, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)


def _scalar(x) -> bool:
    """Truthiness of a possibly-batched terminated/truncated flag."""
    return bool(x[0]) if hasattr(x, "__len__") else bool(x)


def _apply_config(env) -> None:
    """Override demo tuning knobs from --config YAML (configs/demo_tuning.yaml). No-op if unset."""
    if not args_cli.config:
        return
    cfg = yaml.safe_load(Path(args_cli.config).read_text(encoding="utf-8")) or {}
    g = globals()
    for section, mapping in (
        ("controller", {"face_tol": "FACE_TOL", "ang_gain": "ANG_GAIN",
                        "lin_gain": "LIN_GAIN", "deliver_radius": "DELIVER_RADIUS"}),
        ("avoidance", {"influence": "AVOID_INFLUENCE", "push": "AVOID_PUSH",
                       "skip_near_target": "SKIP_NEAR_TARGET"}),
    ):
        for k_yaml, k_glob in mapping.items():
            if k_yaml in cfg.get(section, {}):
                g[k_glob] = float(cfg[section][k_yaml])
    if "grip_radius" in cfg.get("grasp", {}):
        import env.grasp as _grasp
        _grasp.GRIP_RADIUS_M = float(cfg["grasp"]["grip_radius"])   # read live in grasp_success
    sp = cfg.get("spawn", {})
    if "standoff_small" in sp or "standoff_heavy" in sp:
        env._env._spawn_standoff = (float(sp.get("standoff_small", 1.0)),
                                    float(sp.get("standoff_heavy", 1.1)))
    print(f"[demo] applied tuning config {args_cli.config}")


def main() -> None:
    cfg = WarehouseEnvCfg()
    env = WarehouseGymEnv(cfg=cfg)
    env._env.set_stage(args_cli.stage)
    _apply_config(env)   # override tuning knobs from --config (before the first reset/step)
    step_ms = cfg.decimation * cfg.sim.dt * 1000.0   # control period (10 Hz → 100 ms/step)
    print(f"[demo] stage={args_cli.stage}  episodes={args_cli.episodes}  step={step_ms:.0f}ms")

    # Per-step localization log streamed to CSV (flushed each step, so a STUCK/killed run still
    # captures data). Columns documented in the header row.
    fields = ["ep", "step", "t_ms", "x", "y", "yaw_deg", "holding",
              "tgt_x", "tgt_y", "dist", "lin", "ang", "move"]
    csv_file = csv_writer = None
    if args_cli.log_csv:
        path = Path(args_cli.log_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = path.open("w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=fields)
        csv_writer.writeheader()
        csv_file.flush()
        print(f"[demo] streaming localization log → {path}")

    n_rows = 0
    for ep in range(args_cli.episodes):
        obs, _ = env.reset()
        grabbed_at = None
        result = "timeout"
        prev_xy = None
        stuck_count = 0      # consecutive wedged steps
        recover_left = 0     # remaining reverse+turn recovery steps
        backout_left = 0     # remaining straight-reverse steps right after grasping
        for step in range(args_cli.max_steps):
            base_xy = _f(obs["position"])[:2]
            heading = _f(obs["heading"])
            holding = bool(_f(obs["holding"])[0] > 0.5)

            if not holding:
                target = _f(obs["box_pos"])[:2]            # phase A: go to the box
            else:
                if grabbed_at is None:
                    grabbed_at = step
                    backout_left = BACKOUT_STEPS   # reverse out of the grab-rack first
                    print(f"[demo] ep{ep} GRABBED at step {step}")
                target = _f(obs["goal"])[:2]               # phase B: carry to the colour zone

            act = _action(target, base_xy, heading)
            dist, _ = _yaw_err(target, base_xy, heading)
            yaw_deg = math.degrees(math.atan2(heading[1], heading[0]))
            move = 0.0 if prev_xy is None else math.hypot(base_xy[0] - prev_xy[0], base_xy[1] - prev_xy[1])

            # Action override priority: back-out (just grabbed) > recovery (wedged) > steering.
            if backout_left > 0:
                act = np.array([BACKOUT_LIN, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
                backout_left -= 1
                stuck_count = 0
            elif recover_left > 0:
                act = np.array([RECOVER_LIN, RECOVER_ANG, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
                recover_left -= 1
            elif step > 1 and (abs(act[0]) > 0.2 or abs(act[1]) > 0.5) and move < MOVE_EPS:
                # Wedged: commanding motion (drive OR turn) but not moving → reverse+turn to escape.
                stuck_count += 1
                if stuck_count >= RECOVER_AFTER:
                    recover_left, stuck_count = RECOVER_FOR, 0
                    print(f"[demo] ep{ep} WEDGED at step {step} — recovery (reverse+turn)")
            else:
                stuck_count = 0

            row = {
                "ep": ep, "step": step, "t_ms": round(step * step_ms, 1),
                "x": round(float(base_xy[0]), 4), "y": round(float(base_xy[1]), 4),
                "yaw_deg": round(yaw_deg, 1), "holding": int(holding),
                "tgt_x": round(float(target[0]), 4), "tgt_y": round(float(target[1]), 4),
                "dist": round(dist, 4), "lin": round(float(act[0]), 3), "ang": round(float(act[1]), 3),
                "move": round(move, 4),   # base displacement since previous step (0 ≈ stuck)
            }
            if csv_writer is not None:
                csv_writer.writerow(row)
                csv_file.flush()   # survive a stuck/killed run
                n_rows += 1
            if args_cli.log_every and step % args_cli.log_every == 0:
                print(f"[loc] ep{ep} t={row['t_ms']:7.0f}ms step={step:4d} "
                      f"pos=({row['x']:+.2f},{row['y']:+.2f}) yaw={row['yaw_deg']:+6.1f}° "
                      f"hold={row['holding']} tgt=({row['tgt_x']:+.2f},{row['tgt_y']:+.2f}) "
                      f"dist={row['dist']:.2f} act(lin={row['lin']:+.2f},ang={row['ang']:+.2f}) "
                      f"move={row['move']:.3f}")
            prev_xy = base_xy
            obs, reward, terminated, truncated, info = env.step(act)

            if _scalar(terminated):
                result = "DELIVERED (success)" if holding else "terminated (bounds/other)"
                break
            if holding:
                dist_zone, _ = _yaw_err(_f(obs["goal"])[:2], _f(obs["position"])[:2], _f(obs["heading"]))
                if dist_zone < DELIVER_RADIUS:
                    result = "DELIVERED (in zone)"
                    break

        print(f"[demo] ep{ep} result: {result} (grabbed_at={grabbed_at})")

    if csv_file is not None:
        csv_file.close()
        print(f"[demo] wrote {n_rows} localization rows → {args_cli.log_csv}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
