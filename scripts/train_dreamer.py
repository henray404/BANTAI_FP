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
# (docs/project/project_overview.md). Run on a working sim.
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
parser.add_argument("--ca_slope", action="store_true",
                    help="Enable Category-Aware SLOPE reward shaping (configs #4, #6).")
parser.add_argument("--visual_her", action="store_true",
                    help="Enable Visual HER episode relabeling (configs #5, #6).")
parser.add_argument("--config", type=str, default=None,
                    help="Path to experiments/ablation.yaml (tunable hyperparameters).")
parser.add_argument("--stage", type=int, default=3,
                    help="Curriculum stage 1-4 (1=nav/pre-grasped, 2=grasp/spawn-near-box, "
                         "3=full chain [default], 4=full+goal-anneal). Fixed for the run — the "
                         "vendor loop does not auto-advance. Start at 2 to isolate approach+grasp.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# ── cuDNN warm-up BEFORE AppLauncher (vast.ai / RTX 4090 fix) ─────────────────
# On some container + new-driver combos (CUDA 12.9 / driver 575, torch 2.7+cu128), if Omniverse Kit
# grabs the CUDA/Vulkan context first, cuDNN's lazy handle init later fails with
# CUDNN_STATUS_NOT_INITIALIZED at the first F.conv2d (its internal cudaGetDeviceCount sees GPU=NULL)
# even though CUDA itself is healthy. Forcing torch to build the cuDNN handle HERE — before
# AppLauncher launches Kit — sidesteps it, so we keep cuDNN enabled (fast) instead of the slow
# torch.backends.cudnn.enabled=False workaround. No-op without CUDA. See the cuDNN-init bug note.
try:
    import torch

    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        _ = torch.nn.functional.conv2d(
            torch.zeros(1, 1, 8, 8, device="cuda"),
            torch.zeros(1, 1, 3, 3, device="cuda"),
        )
        torch.cuda.synchronize()
        print("[train_dreamer] cuDNN warm-up OK (handle created before AppLauncher).")
except Exception as exc:  # never block training on the warm-up — fall through to AppLauncher
    print(f"[train_dreamer] cuDNN warm-up skipped: {exc}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Project imports (after AppLauncher) ───────────────────────────────
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from env.warehouse_env import WarehouseEnvCfg, WarehouseGymEnv  # noqa: E402
from reward.ca_slope import CASlopeShaper  # noqa: E402
from reward.ca_slope_wrapper import CASlopeEnvWrapper  # noqa: E402
from experiments.metrics import BestModelTracker, EvalCsv, write_run_config  # noqa: E402
from experiments.settings import load_settings  # noqa: E402
from experiments.trajectory_recorder import TrajectoryRecorder  # noqa: E402
from models.dreamerv3.config import add_vendor_to_path, build_config  # noqa: E402
from models.dreamerv3.warehouse_dreamer_env import make_warehouse_dreamer  # noqa: E402

# Resolved tunable settings (ablation.yaml over defaults). Used to build CA-SLOPE / HER /
# eval cadence so the YAML actually controls the run.
_SETTINGS = load_settings(args_cli.config)

# One shared env (single Isaac sim). `gym` is the gym env (maybe CA-SLOPE-wrapped);
# `success` is the raw WarehouseGymEnv used for success readout in eval.
_SHARED_ENV = {"gym": None, "success": None}
_EVAL = {"csv": None, "best": None, "traj": None}  # set in main(); eval logs + best + trace.


def _build_shared_env():
    """Build the single env (num_envs=1) once and cache it; wrap with CA-SLOPE if asked."""
    if _SHARED_ENV["gym"] is None:
        cfg = WarehouseEnvCfg()
        cfg.scene.num_envs = 1
        raw = WarehouseGymEnv(cfg=cfg)
        # Curriculum stage (fixed for the run; vendor loop has no success-gate to auto-advance).
        # set_stage validates 1-4 and applies on the next reset (the dreamer loop resets before
        # stepping). _env is the underlying WarehouseRLEnv exposing the curriculum API.
        raw._env.set_stage(args_cli.stage)
        print(f"[curriculum] stage fixed at {args_cli.stage}", flush=True)
        _SHARED_ENV["success"] = raw
        if args_cli.ca_slope:
            cs = _SETTINGS.ca_slope
            shaper = CASlopeShaper(
                gamma=cs["gamma"],
                category_gains=tuple(cs["category_gains"]),
                generic_gain=cs["generic_gain"],
                phase_b_offset=cs["phase_b_offset"],
                category_aware=(cs["mode"] == "category"),
            )
            _SHARED_ENV["gym"] = CASlopeEnvWrapper(raw, shaper=shaper, mode=cs["mode"])
        else:
            _SHARED_ENV["gym"] = raw
    return _SHARED_ENV["gym"]


def main() -> None:
    """Patch the vendor make_env and run NM512's Dreamer training loop."""
    add_vendor_to_path()
    # The vendor uses bare `import models` (NM512 runs from its own dir). The project `models`
    # package is already in sys.modules (models.dreamerv3.*) and shadows vendor/models.py, so the
    # vendor's `import models` returns the project package (no WorldModel). Temporarily evict the
    # project `models.*` entries so `import dreamer` binds the vendor module globals to vendor/*.py
    # (sys.path[0]), then restore them — later code (make_warehouse_dreamer) still does
    # `from models.dreamerv3.config import ...` and needs the project package back.
    _saved = {k: sys.modules.pop(k)
              for k in [m for m in sys.modules if m == "models" or m.startswith("models.")]}
    try:
        import dreamer  # vendored top-level entry module; binds its `models` -> vendor/models.py
    finally:
        sys.modules.update(_saved)

    # Visual HER (configs #5, #6): inject relabeled episodes into the train cache.
    if args_cli.visual_her:
        import tools  # vendored top-level module (on sys.path after add_vendor_to_path)
        from env.her_nm512 import install_visual_her
        her = _SETTINGS.visual_her
        install_visual_her(her_ratio=her["her_ratio"],
                           success_reward=her["success_reward"], seed=args_cli.seed)
        _ = tools  # keep the import referenced (install patches tools.* by name)

    _EVAL["csv"] = EvalCsv(args_cli.logdir)
    # Reproducibility: snapshot exactly what this run used; track the best checkpoint.
    write_run_config(args_cli.logdir, {
        "algo": "dreamer", "seed": args_cli.seed, "steps": args_cli.steps,
        "ca_slope": args_cli.ca_slope, "visual_her": args_cli.visual_her,
        "config_file": args_cli.config, "settings": vars(_SETTINGS),
    })
    _EVAL["best"] = BestModelTracker(args_cli.logdir)
    _EVAL["traj"] = TrajectoryRecorder(_EVAL["best"].dir)  # best-episode action trace

    b, dm = _SETTINGS.budget, _SETTINGS.dreamer
    config = build_config(
        extra_overrides={"seed": args_cli.seed, "steps": args_cli.steps,
                         "eval_every": b["eval_every"],
                         "eval_episode_num": b["eval_episodes"],
                         "train_ratio": dm["train_ratio"], "prefill": dm["prefill"]},
        logdir=args_cli.logdir,
    )

    # Replace suite dispatch with our warehouse env (shared single sim). Eval envs get an
    # EvalRecorder so DreamerV3 logs the same success-rate CSV as the SAC/PPO baselines.
    def _make_env(cfg, mode, env_id):
        env = make_warehouse_dreamer(_build_shared_env(), cfg)
        if mode == "eval":
            from experiments.nm512_eval import EvalRecorder
            env = EvalRecorder(
                env, _SHARED_ENV["success"], _EVAL["csv"],
                eval_episodes=cfg.eval_episode_num, eval_every=cfg.eval_every,
                best=_EVAL["best"],
                checkpoint_src=Path(args_cli.logdir) / "latest.pt",
                traj=_EVAL["traj"],
            )
        return env

    dreamer.make_env = _make_env
    dreamer.main(config)


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # noqa: BLE001 — diagnostic: surface the real error before close() hides it
        import traceback
        print("[DIAG] main() raised:", flush=True)
        traceback.print_exc()
        raise
    finally:
        if _SHARED_ENV["gym"] is not None:
            _SHARED_ENV["gym"].close()
        simulation_app.close()
