# scripts/train_dreamer.py — DreamerV3 (vanilla, NM512) on WarehouseGymEnv.
#
# ENTRY SCRIPT: owns AppLauncher (env/ + adapter modules must not — see
# bugs_errors/2026-05-15_double-applaunch-crash.md).
#
# Strategy: reuse the vendored NM512 training loop (dreamer.main) but monkeypatch
# its make_env to return our WarehouseDreamer wrapper around a SINGLE shared
# WarehouseGymEnv (one Isaac sim per process). Train + eval share that sim — fine
# for a first learning-curve smoke; for clean eval numbers run a dedicated process.
#
# UNVERIFIED on this hardware: env needs `pixels`; Blackwell camera SDP blocker has
# kept the full env from running end-to-end on the RTX 5050
# (docs/project_overview.md). Run on a working sim.
#
# Deps: vendored NM512 (models/dreamerv3/vendor) + its requirements (gym==0.22,
# ruamel.yaml, einops, ...). See requirements-ml.txt. Do NOT let these downgrade the
# pinned isaaclab torch 2.7.0+cu128.
#
# Usage:
#   python scripts/train_dreamer.py --num_envs 1
#   python scripts/train_dreamer.py --headless

"""Train DreamerV3 (vanilla) on the warehouse nav task via the vendored NM512 loop."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train DreamerV3 on WarehouseGymEnv")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--steps", type=int, default=200000, help="Total env steps")
parser.add_argument("--logdir", type=str, default="training/results/dreamerv3")
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
from models.dreamerv3.config import add_vendor_to_path, build_config  # noqa: E402
from models.dreamerv3.warehouse_dreamer_env import make_warehouse_dreamer  # noqa: E402

# One shared WarehouseGymEnv (single Isaac sim). Built lazily on first make_env call.
_SHARED_ENV = {"env": None}


def _build_shared_env() -> WarehouseGymEnv:
    """Build the single WarehouseGymEnv (num_envs=1) once and cache it."""
    if _SHARED_ENV["env"] is None:
        cfg = WarehouseEnvCfg()
        cfg.scene.num_envs = 1
        _SHARED_ENV["env"] = WarehouseGymEnv(cfg=cfg)
    return _SHARED_ENV["env"]


def main() -> None:
    """Patch the vendor make_env and run NM512's Dreamer training loop."""
    add_vendor_to_path()
    import dreamer  # vendored top-level entry module

    config = build_config(
        extra_overrides={"seed": args_cli.seed, "steps": args_cli.steps},
        logdir=args_cli.logdir,
    )

    # Replace suite dispatch with our warehouse env (shared single sim).
    def _make_env(cfg, mode, env_id):
        return make_warehouse_dreamer(_build_shared_env(), cfg)

    dreamer.make_env = _make_env
    dreamer.main(config)


if __name__ == "__main__":
    try:
        main()
    finally:
        if _SHARED_ENV["env"] is not None:
            _SHARED_ENV["env"].close()
        simulation_app.close()
