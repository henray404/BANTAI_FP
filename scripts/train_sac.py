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
parser.add_argument("--ca_slope", action="store_true",
                    help="Enable Category-Aware SLOPE reward shaping (off for #1/#2).")
parser.add_argument("--config", type=str, default=None,
                    help="Path to experiments/ablation.yaml (hyperparams + eval cadence).")
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

    from experiments.settings import load_settings
    from experiments.metrics import write_run_config
    settings = load_settings(args_cli.config)
    # Reproducibility: snapshot exactly what this run used.
    write_run_config(logdir, {
        "algo": args_cli.algo, "seed": args_cli.seed, "timesteps": args_cli.timesteps,
        "ca_slope": args_cli.ca_slope, "config_file": args_cli.config,
        "settings": vars(settings),
    })

    cfg = WarehouseEnvCfg()
    cfg.scene.num_envs = 1
    warehouse = WarehouseGymEnv(cfg=cfg)
    if args_cli.ca_slope:
        from reward.ca_slope import CASlopeShaper
        from reward.ca_slope_wrapper import CASlopeEnvWrapper
        cs = settings.ca_slope
        shaper = CASlopeShaper(
            gamma=cs["gamma"], category_gains=tuple(cs["category_gains"]),
            generic_gain=cs["generic_gain"], phase_b_offset=cs["phase_b_offset"],
            category_aware=(cs["mode"] == "category"),
        )
        warehouse = CASlopeEnvWrapper(warehouse, shaper=shaper, mode=cs["mode"])
    # Monitor records episode reward/length into the SB3 logger -> TB gets
    # rollout/ep_rew_mean + rollout/ep_len_mean (without it the ep_info_buffer stays empty
    # and only train/* losses are logged — no way to see if episodes collapse to length ~5).
    # Eval uses the INNER env (bypasses Monitor) so eval episodes don't pollute train stats.
    from stable_baselines3.common.monitor import Monitor
    inner_env = SB3WarehouseEnv(warehouse)
    sb3_env = Monitor(inner_env)

    if args_cli.algo == "sac":
        from training.baselines.sac import build_sac

        model = build_sac(sb3_env, seed=args_cli.seed, tensorboard_log=str(logdir),
                          **settings.algo_kwargs("sac"))
    else:
        from training.baselines.ppo import build_ppo

        model = build_ppo(sb3_env, seed=args_cli.seed, tensorboard_log=str(logdir),
                          **settings.algo_kwargs("ppo"))

    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from experiments.metrics import BestModelTracker, EvalCsv, evaluate_policy
    from experiments.trajectory_recorder import TrajectoryRecorder

    class EvalSuccessCallback(BaseCallback):
        """Every eval_every steps: eval → eval_metrics.csv; save best model + action trace."""

        def __init__(self, eval_env, csv: EvalCsv, best: BestModelTracker,
                     traj: TrajectoryRecorder, every: int, episodes: int):
            """Hold the eval env + CSV writer + best tracker + trajectory recorder + cadence."""
            super().__init__()
            self._eval_env = eval_env
            self._csv = csv
            self._best = best
            self._traj = traj
            self._every = every
            self._episodes = episodes
            self._next = every

        def _on_step(self) -> bool:
            """Trigger an eval when the step counter crosses the next boundary."""
            if self.num_timesteps >= self._next:
                self._next += self._every
                act = lambda obs: self.model.predict(obs, deterministic=True)[0]
                metrics = evaluate_policy(self._eval_env, act, self._episodes,
                                          recorder=self._traj)
                self._csv.log(self.num_timesteps, metrics)
                # Mirror eval metrics into TB (eval/*) so the staged success shows on graphs.
                for k in ("success_rate", "grasp_rate", "reach_rate",
                          "mean_length", "mean_return"):
                    if k in metrics:
                        self.logger.record(f"eval/{k}", metrics[k])
                if self._best.update(self.num_timesteps, metrics):
                    self.model.save(str(self._best.dir / "best_model"))
                # BUG 3 resync: eval reset+stepped the SAME underlying Isaac env (only one sim
                # per process), desyncing SB3's cached rollout obs. Start a fresh episode here so
                # the trainer does not bootstrap value across the eval gap / train on stale obs.
                import numpy as _np
                self.model._last_obs = self.model.env.reset()
                self.model._last_episode_starts = _np.ones(
                    (self.model.env.num_envs,), dtype=bool)
            return True

    _best = BestModelTracker(logdir)
    callbacks = [EvalSuccessCallback(
        inner_env, EvalCsv(logdir), _best, TrajectoryRecorder(_best.dir),
        settings.budget["eval_every"], settings.budget["eval_episodes"])]
    if args_cli.wandb:
        try:
            from wandb.integration.sb3 import WandbCallback
            import wandb

            wandb.init(project="bantai-warehouse", name=f"{args_cli.algo}_seed{args_cli.seed}",
                       config={"algo": args_cli.algo, "seed": args_cli.seed},
                       sync_tensorboard=True)
            callbacks.append(WandbCallback())
        except ImportError:
            print("[train] wandb requested but not installed — skipping W&B.")

    try:
        model.learn(total_timesteps=args_cli.timesteps, callback=CallbackList(callbacks))
        model.save(str(logdir / f"{args_cli.algo}_seed{args_cli.seed}"))
        print(f"[train] saved → {logdir}/{args_cli.algo}_seed{args_cli.seed}.zip")
    finally:
        warehouse.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
