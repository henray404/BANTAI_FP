# scripts/train_p3.py — P3 DreamerV3 + Visual HER training entry point.
#
# ENTRY SCRIPT: owns AppLauncher (see bugs_errors/2026-05-15_double-applaunch-crash.md).
# All imports from env/ and policy/ happen AFTER AppLauncher.
#
# REQUIRES:
#   P1: WarehouseEnvCfg with obs_v2 (9 keys) + action (6,) wired in.
#       Current warehouse_env.py still uses the old (2,) contract.
#       Run this script once P1 commits obs_v2 + action_pickup changes.
#   P2: A WorldModelInterface implementation (models/dreamerv3/__init__.py).
#       Until P2 delivers, run with --mock_wm to test the P3 loop in isolation.
#
# Usage:
#   python scripts/train_p3.py --headless
#   python scripts/train_p3.py --num_envs 1 --steps 200000 --seed 0
#   python scripts/train_p3.py --headless --mock_wm   # CPU test, no P2 needed

"""Train P3 DreamerV3 actor-critic + Visual HER on the warehouse pickup task."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Train P3: DreamerV3 + Visual HER")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--steps", type=int, default=200_000)
parser.add_argument("--logdir", type=str, default="training/results/p3")
parser.add_argument("--wandb_mode", type=str, default="online",
                    choices=["online", "offline", "disabled"])
parser.add_argument("--mock_wm", action="store_true",
                    help="Use dummy WorldModel (no P2 needed). Dev/testing only.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True   # pixels obs requires camera flag

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Project imports (after AppLauncher) ──────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch                                                      # noqa: E402
from env.warehouse_env import WarehouseEnvCfg, WarehouseGymEnv   # noqa: E402
from policy.config import P3Config                               # noqa: E402
from policy.train_loop import P3Trainer, WorldModelInterface     # noqa: E402


# ── Mock world model (P3 dev only, replace with P2's implementation) ─────────

class _MockWorldModel(WorldModelInterface):
    """Dummy world model returning random RSSM features.

    Use with --mock_wm to test the P3 training loop before P2 is ready.
    DO NOT use for actual experiments.
    """

    _FEAT_DIM = 1536  # must match P3Config.feat_dim

    def get_feat_dim(self) -> int:
        return self._FEAT_DIM

    def encode_obs(self, obs: dict, device: str = "cuda:0") -> torch.Tensor:
        vals = list(obs.values())
        B = vals[0].shape[0] if hasattr(vals[0], "shape") else 1
        return torch.randn(B, self._FEAT_DIM, device=device)

    def imagine_step(
        self, feat: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, dev = feat.shape[0], feat.device
        next_feat = feat + 0.01 * torch.randn_like(feat)
        return next_feat, torch.zeros(B, device=dev), torch.ones(B, device=dev)

    def train_batch(self, batch, device: str = "cuda:0") -> dict[str, float]:
        return {"wm/loss": 0.0, "wm/kl": 0.0}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Build env, world model, and P3 trainer; run training loop."""

    # Build Isaac Lab env (obs_v2 + action (6,) required from P1).
    env_cfg = WarehouseEnvCfg()
    env_cfg.scene.num_envs = 1
    env = WarehouseGymEnv(cfg=env_cfg)

    # World model: mock until P2 delivers.
    if args_cli.mock_wm:
        print("[train_p3] WARNING: _MockWorldModel active (--mock_wm). "
              "Replace with P2's WarehouseWorldModel for real experiments.")
        world_model = _MockWorldModel()
    else:
        # P2 exposes: from models.dreamerv3 import WarehouseWorldModel
        try:
            from models.dreamerv3 import WarehouseWorldModel  # type: ignore[import]
            world_model = WarehouseWorldModel(
                obs_space=env.observation_space,
                action_dim=6,
                device=args_cli.device if hasattr(args_cli, "device") else "cuda:0",
            )
        except ImportError as exc:
            raise ImportError(
                "P2's WarehouseWorldModel not found in models/dreamerv3/__init__.py.\n"
                "Run with --mock_wm to test P3 without P2, or wait for P2 to deliver.\n"
                f"Original error: {exc}"
            )

    p3_cfg = P3Config(
        seed=args_cli.seed,
        logdir=args_cli.logdir,
        wandb_mode=args_cli.wandb_mode,
    )

    trainer = P3Trainer(env=env, world_model=world_model, cfg=p3_cfg)

    try:
        trainer.run(total_steps=args_cli.steps)
    finally:
        ckpt = trainer.save()
        print(f"[train_p3] Checkpoint saved → {ckpt}")
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
