# run_env.py — Entry point for the warehouse robot environment.
#
# Usage:
#   python scripts/run_env.py --num_envs 1                 # windowed, single env
#   python scripts/run_env.py --num_envs 4 --headless      # headless, 4 envs
#
# AppLauncher MUST be created here, before any isaaclab imports from env/.
# See bugs_errors/2026-05-15_double-applaunch-crash.md.

"""Launch and exercise the warehouse env with a random policy."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

# ── CLI + AppLauncher ─────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Run warehouse robot environment")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel envs")
parser.add_argument("--steps", type=int, default=200, help="Total env steps to run")
parser.add_argument("--print_every", type=int, default=20, help="Print reward stats every N steps")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # warehouse env always uses onboard camera

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Project imports (after AppLauncher is live) ───────────────────────
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from env.warehouse_env import WarehouseEnvCfg, WarehouseGymEnv  # noqa: E402


def random_policy_loop(env: WarehouseGymEnv, total_steps: int, print_every: int) -> None:
    """Drive the env with random [linear, angular] actions; print reward stats."""
    obs, _ = env.reset()
    print(f"[OK] reset complete. obs keys = {list(obs.keys())}")
    for key, val in obs.items():
        print(f"      {key}: shape={tuple(val.shape)}, dtype={val.dtype}")

    cumulative = torch.zeros(env.num_envs, device=env.device)
    for step in range(total_steps):
        if not simulation_app.is_running():
            break
        action = np.random.uniform(-1.0, 1.0, size=(env.num_envs, 2)).astype(np.float32)
        obs, reward, terminated, truncated, _ = env.step(action)
        cumulative += reward
        if (step + 1) % print_every == 0:
            done = (terminated | truncated).sum().item()
            print(
                f"[step {step+1:04d}] "
                f"reward mean={reward.mean().item():+.4f} "
                f"cum mean={cumulative.mean().item():+.4f} "
                f"done={done}/{env.num_envs}"   
            )


def main() -> None:
    """Build env, run random policy, close cleanly."""
    cfg = WarehouseEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    env = WarehouseGymEnv(cfg=cfg)
    try:
        random_policy_loop(env, total_steps=args_cli.steps, print_every=args_cli.print_every)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
