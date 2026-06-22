# scripts/record_scenario.py — run a scenario in the REAL env and record EVERYTHING to CSV + meta.
#
# Captures, per control step: all joint positions/velocities (kinematics), base_link world pose
# (xyz+quat+rpy), end-effector pose, target box pose, gripper/holding/grasp events, goal, action,
# reward, contact force, termination. Plus a <name>.meta.json with the full scenario + env config so
# the run is reconstructable and replayable (see scripts/replay_csv.py).
#
# Must run in the Isaac env (Windows box), NOT a Mac:
#   conda activate isaaclab
#   python scripts/record_scenario.py --seed 0 --policy random --out runs/heavy_seed0
#   python scripts/record_scenario.py --seed 0 --policy checkpoint --ckpt training/results/p3.pt \
#       --slope category --out runs/best_run
#
# AppLauncher MUST be created before any isaaclab imports (see bugs_errors/2026-05-15...).

"""Record a full scenario rollout (all joints + poses + events + metadata) to CSV for replay."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record a full scenario rollout to CSV + meta.json")
parser.add_argument("--out", default="runs/run", help="output run path (no extension)")
parser.add_argument("--seed", type=int, default=0, help="env seed (also stored in meta for replay)")
parser.add_argument("--steps", type=int, default=1000, help="max control steps to record")
parser.add_argument("--policy", choices=["random", "checkpoint"], default="random",
                    help="random actions, or load a P3 actor checkpoint")
parser.add_argument("--ckpt", default="", help="P3 checkpoint path (for --policy checkpoint)")
parser.add_argument("--slope", choices=["category", "generic", "none"], default="none",
                    help="wrap env with CA-SLOPE shaping and record the shaping term per step")
parser.add_argument("--stop_on_success", type=int, default=1, help="stop recording when delivered")
parser.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="record MULTIPLE runs (one per seed) to <out>/seed<n>, ready for "
                         "scripts/rank_runs.py per-scenario ranking. Overrides single --seed/--out.")
parser.add_argument("--checkpoints", type=int, default=1,
                    help="auto state-checkpoints + rewind on idle/spin/collision (1=on)")
parser.add_argument("--idle_seconds", type=float, default=30.0,
                    help="rewind to nearest checkpoint if the robot is stuck this long")
parser.add_argument("--progress_delta", type=float, default=2.5,
                    help="snapshot when the robot gets this many metres CLOSER to its active target "
                         "(progress-based, not a step counter — no checkpoint while moving away). "
                         "2.5m suits the 20x30m warehouse.")
parser.add_argument("--checkpoint_every", type=int, default=0,
                    help="optional fixed-interval fallback snapshot every N steps (0 = off)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from env.warehouse_env import WarehouseEnvCfg, WarehouseGymEnv  # noqa: E402
from recording.recorder import TrajectoryRecorder  # noqa: E402
from recording.state_extractor import _contact_force, _inner, _rpy_deg, build_metadata, step_row  # noqa: E402


def _checkpoint_signals(env) -> dict:
    """Per-step signals the CheckpointManager needs, all in one consistent (world) frame."""
    ie = _inner(env)
    robot = ie.scene["robot"]
    blink = robot.body_names.index("base_link")
    base_p = robot.data.body_pos_w[0, blink]
    _, _, yaw_deg = _rpy_deg(robot.data.body_quat_w[0, blink])
    origin = ie.scene.env_origins[0]
    box = ie.scene[ie.target_box_name[0]]
    base_xy = (float(base_p[0]), float(base_p[1]))
    box_xy = (float(box.data.root_pos_w[0, 0]), float(box.data.root_pos_w[0, 1]))
    goal_xy = (float(ie.goal_pos[0, 0] + origin[0]), float(ie.goal_pos[0, 1] + origin[1]))
    local_x, local_y = base_xy[0] - float(origin[0]), base_xy[1] - float(origin[1])
    return {
        "base_xy": base_xy,
        "yaw": math.radians(yaw_deg),
        "holding": bool(ie.holding[0].item()),
        "dist_box": math.hypot(base_xy[0] - box_xy[0], base_xy[1] - box_xy[1]),
        "dist_goal": math.hypot(box_xy[0] - goal_xy[0], box_xy[1] - goal_xy[1]),
        "contact_force": _contact_force(ie),
        # room interior half-extents match env.out_of_bounds (9.5 x 14.5 m -> ~20x30 m warehouse).
        "out_of_bounds": abs(local_x) > 9.5 or abs(local_y) > 14.5,
    }


def _make_policy(env):
    """Return a callable obs -> action(6,). Random by default; loads a P3 actor if requested."""
    if args_cli.policy == "random":
        return lambda obs: np.random.uniform(-1.0, 1.0, size=(env.num_envs, 6)).astype(np.float32)

    from policy.actor_critic import Actor  # lazy: only needed for checkpoint
    from policy.config import P3Config
    cfg = P3Config()
    actor = Actor(feat_dim=cfg.feat_dim, action_dim=cfg.action_dim, hidden=cfg.actor_hidden).to(env.device)
    ckpt = torch.load(args_cli.ckpt, map_location=env.device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    # NOTE: a real DreamerV3 actor needs the RSSM feature, not raw obs. Until P2's world model is
    # wired, prefer --policy random for recording. This branch is a stub for when the actor lands.
    raise NotImplementedError(
        "checkpoint replay needs the P2 world-model feature encoder (not ready). Use --policy random "
        "for now, or record from teleop via scripts/drive_env.py."
    )


def record_one(env, policy, seed: int, out_stem) -> dict:
    """Record one episode (reset(seed) -> rollout) to <out_stem>.csv + .meta.json. Returns summary."""
    obs, _ = env.reset(seed=seed)
    meta = build_metadata(env, seed=seed, policy=args_cli.policy,
                          run_id=Path(out_stem).name, extra={"slope_mode": args_cli.slope})
    rec = TrajectoryRecorder(out_stem, metadata=meta)
    print(f"[record] seed {seed}: {meta['category']} box {meta['target_box_name']} "
          f"-> zone {meta['goal_zone_xyz']}")

    ckpt_mgr = None
    if args_cli.checkpoints:
        from recording.checkpoint import CheckpointManager
        from recording.sim_state import capture_sim_state, restore_sim_state
        ckpt_mgr = CheckpointManager(
            capture_fn=capture_sim_state, restore_fn=restore_sim_state,
            control_hz=meta["control_hz"], idle_seconds=args_cli.idle_seconds,
            progress_delta=args_cli.progress_delta, period_steps=args_cli.checkpoint_every,
        )
        ckpt_mgr.reset(env)

    t0 = time.time()
    success = False
    grasp_step = deliver_step = -1
    total_r = 0.0
    n_rewinds = 0
    step = 0
    try:
        for step in range(args_cli.steps):
            if not simulation_app.is_running():
                break
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            slope = info.get("ca_slope_shaping") if isinstance(info, dict) else None
            slope_v = float(np.asarray(slope.tolist() if hasattr(slope, "tolist") else slope).reshape(-1)[0]) \
                if slope is not None else None

            row = step_row(env, step=step, t=time.time() - t0, action=action, reward=reward,
                           terminated=terminated, truncated=truncated, info=info, slope_reward=slope_v)

            # Checkpoint milestones + auto-rewind on idle/spin/collision; annotate the row.
            ck_label, restore_reason = "", ""
            if ckpt_mgr is not None:
                ev = info if isinstance(info, dict) else {}
                out = ckpt_mgr.observe(env, step=step,
                                       grasp_event=bool(ev.get("grasp_event", False)),
                                       drop_event=bool(ev.get("drop_event", False)),
                                       **_checkpoint_signals(env))
                ck_label = out.snapshot_label or ""
                if out.restored:
                    restore_reason = out.reason
                    n_rewinds += 1
                    # sim state already rewound by restore_sim_state; next env.step() yields fresh obs.
                    print(f"[record] step {step}: rewind ({out.reason}) -> checkpoint @ step {out.checkpoint_step}")
            row["checkpoint_event"] = ck_label
            row["restore_reason"] = restore_reason
            rec.add(row)

            total_r += float(np.asarray(reward.tolist() if hasattr(reward, "tolist") else reward).reshape(-1)[0])
            if isinstance(info, dict):
                if info.get("grasp_event") and grasp_step < 0:
                    grasp_step = step
                if info.get("deliver_event") and deliver_step < 0:
                    deliver_step = step
                    success = True
            done = bool(np.asarray(terminated.tolist() if hasattr(terminated, "tolist") else terminated).reshape(-1)[0]) \
                or bool(np.asarray(truncated.tolist() if hasattr(truncated, "tolist") else truncated).reshape(-1)[0])
            if done or (success and args_cli.stop_on_success):
                break
    finally:
        summary = {"success": int(success), "return": round(total_r, 4), "steps": step + 1,
                   "grasp_step": grasp_step, "deliver_step": deliver_step, "n_rewinds": n_rewinds}
        rec.set_summary(summary)
        rec.close()
    print(f"[record] wrote {rec.csv_path} ({step + 1} steps) | success={success} "
          f"return={total_r:.2f} rewinds={n_rewinds}")
    return summary


def main() -> None:
    """Build env+policy once, then record one run per seed (per-scenario batch for ranking)."""
    cfg = WarehouseEnvCfg()
    cfg.scene.num_envs = 1
    env = WarehouseGymEnv(cfg=cfg)

    if args_cli.slope != "none":
        from reward.ca_slope_wrapper import CASlopeEnvWrapper
        env = CASlopeEnvWrapper(env, mode=args_cli.slope)

    policy = _make_policy(env)
    try:
        if args_cli.seeds:  # batch: one run per seed -> <out>/seed<n> (rank with rank_runs.py)
            for seed in args_cli.seeds:
                record_one(env, policy, seed, str(Path(args_cli.out) / f"seed{seed}"))
            print(f"[record] batch done. Rank: python scripts/rank_runs.py --dir {args_cli.out}")
        else:               # single run -> <out>
            record_one(env, policy, args_cli.seed, args_cli.out)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
