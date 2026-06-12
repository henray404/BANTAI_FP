# scripts/train_sac.py — SAC / PPO baseline on WarehouseGymEnv (stable-baselines3).
#
# ENTRY SCRIPT: owns AppLauncher. Wraps WarehouseGymEnv (num_envs=1) via
# training.env_adapter.SB3WarehouseEnv → DummyVecEnv → SB3 SAC (MultiInputPolicy).
#
# UNVERIFIED on this hardware (Blackwell camera blocker). Run on a working sim.
# See training/env_adapter.py for the Isaac auto-reset caveat at episode boundaries.
#
# Usage:
#   python scripts/train_sac.py --algo sac --timesteps 200000
#   python scripts/train_sac.py --algo ppo --timesteps 500000 --headless

"""Train an SB3 SAC/PPO baseline on the warehouse nav task."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train SAC/PPO baseline on WarehouseGymEnv")
parser.add_argument("--algo", choices=["sac", "ppo"], default="sac")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--timesteps", type=int, default=200000)
parser.add_argument("--logdir", type=str, default="training/results/baseline")
parser.add_argument("--wandb", action="store_true", help="Log to W&B if installed")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Project imports (after AppLauncher) ───────────────────────────────
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from env.warehouse_env import WarehouseEnvCfg, WarehouseGymEnv  # noqa: E402
from training.env_adapter import SB3WarehouseEnv  # noqa: E402
from training.seed import seed_everything  # noqa: E402


def main() -> None:
    """Build env + SB3 agent, train, save."""
    seed_everything(args_cli.seed)
    logdir = Path(args_cli.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    cfg = WarehouseEnvCfg()
    cfg.scene.num_envs = 1
    warehouse = WarehouseGymEnv(cfg=cfg)
    sb3_env = SB3WarehouseEnv(warehouse)

    if args_cli.algo == "sac":
        from training.baselines.sac import build_sac

        model = build_sac(sb3_env, seed=args_cli.seed, tensorboard_log=str(logdir))
    else:
        from training.baselines.ppo import build_ppo

        model = build_ppo(sb3_env, seed=args_cli.seed, tensorboard_log=str(logdir))

    callback = None
    if args_cli.wandb:
        try:
            from wandb.integration.sb3 import WandbCallback
            import wandb

            wandb.init(project="bantai-warehouse", name=f"{args_cli.algo}_seed{args_cli.seed}",
                       config={"algo": args_cli.algo, "seed": args_cli.seed},
                       sync_tensorboard=True)
            callback = WandbCallback()
        except ImportError:
            print("[train] wandb requested but not installed — skipping W&B.")

    try:
        model.learn(total_timesteps=args_cli.timesteps, callback=callback)
        model.save(str(logdir / f"{args_cli.algo}_seed{args_cli.seed}"))
        print(f"[train] saved → {logdir}/{args_cli.algo}_seed{args_cli.seed}.zip")
    finally:
        warehouse.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
