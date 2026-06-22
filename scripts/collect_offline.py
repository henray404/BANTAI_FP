# scripts/collect_offline.py — collect a replay dataset LOCALLY (needs Isaac Sim).
#
# Runs the warehouse env and saves NM512-format .npz episodes to --out. Zip that
# dir, upload to Colab, and train with scripts/train_offline.py (no sim needed).
#
# ENTRY SCRIPT: owns AppLauncher (see bugs_errors/2026-05-15_double-applaunch-crash.md).
# All env/ and vendor imports happen AFTER AppLauncher.
#
# ponytail: random policy. Enough to teach the world model dynamics; for data that
# also teaches good behavior, swap _random_agent for a scripted approach policy.
#
# Usage:
#   python scripts/collect_offline.py --steps 20000 --out training/data/episodes --headless

"""Collect NM512 replay episodes from the warehouse sim for offline training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect offline replay episodes (needs sim)")
parser.add_argument("--steps", type=int, default=20_000, help="Total env steps to collect")
parser.add_argument("--out", type=str, default="training/data/episodes")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--stage", type=int, default=3,
                    help="Curriculum stage 1-4 (see scripts/train_dreamer.py).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # pixels obs requires the camera flag

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Project imports (after AppLauncher) ────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch                                                          # noqa: E402
from torch import distributions as torchd                            # noqa: E402

from env.warehouse_env import WarehouseEnvCfg, WarehouseGymEnv       # noqa: E402
from models.dreamerv3.config import add_vendor_to_path, build_config  # noqa: E402
from models.dreamerv3.warehouse_dreamer_env import make_warehouse_dreamer  # noqa: E402


def _random_agent(action_space):
    """Uniform random policy over the action box, NM512 agent signature."""
    actor = torchd.independent.Independent(
        torchd.uniform.Uniform(
            torch.tensor(action_space.low).repeat(1, 1),
            torch.tensor(action_space.high).repeat(1, 1)),
        1,
    )

    def agent(obs, done, state):
        """Return ({'action','logprob'}, state) — ignores obs (random)."""
        a = actor.sample()
        return {"action": a, "logprob": actor.log_prob(a)}, None

    return agent


def main() -> None:
    """Build the sim env, wrap it NM512-style, and simulate a random policy to disk."""
    add_vendor_to_path()
    config = build_config(extra_overrides={"seed": args_cli.seed}, logdir="training/data")

    out = Path(args_cli.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = WarehouseEnvCfg()
    cfg.scene.num_envs = 1
    raw = WarehouseGymEnv(cfg=cfg)
    raw._env.set_stage(args_cli.stage)
    print(f"[collect] curriculum stage {args_cli.stage}", flush=True)
    env = make_warehouse_dreamer(raw, config)

    import tools  # vendor/tools.py (VENDOR_DIR on sys.path via add_vendor_to_path)
    from parallel import Damy

    envs = [Damy(env)]
    logger = tools.Logger(out, 0)
    agent = _random_agent(envs[0].action_space)

    print(f"[collect] simulating {args_cli.steps} steps → {out}", flush=True)
    tools.simulate(agent, envs, {}, out, logger, steps=args_cli.steps)
    n = len(list(out.glob("*.npz")))
    print(f"[collect] done — {n} episodes in {out}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # noqa: BLE001 — surface the real error before close() hides it
        import traceback
        print("[DIAG] collect_offline raised:", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
