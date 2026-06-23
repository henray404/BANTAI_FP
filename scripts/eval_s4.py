# scripts/eval_s4.py — s4 TRANSFER TEST: eval the best trained model on the modified s4 scene.
#
# ENTRY SCRIPT: owns AppLauncher. Builds WarehouseGymEnv with the s4 scene override
# (configs/env_config_s4.yaml: physically smaller rack + one target box on the 2nd shelf level),
# loads a model trained on the NORMAL scene (s1 PPO / s2 DreamerV3 / s3 DreamerV3+CA-SLOPE), and
# runs deterministic eval episodes. NO training happens here — this measures generalization to a
# scene the model never saw.
#
# The override is wired by setting $WAREHOUSE_ENV_CONFIG BEFORE importing the env, so
# env/warehouse_scene reads the s4 `scene:` block (rack scale + second_shelf_box) at import time.
#
# UNVERIFIED on this hardware (Blackwell camera blocker) — run on the Linux/A100 box.
#
# Usage:
#   # auto-pick the best run across s1-s3 results, eval on s4:
#   python scripts/eval_s4.py --auto --results training/results/ablation
#   # or point at one run's best/ dir explicitly:
#   python scripts/eval_s4.py --run training/results/ablation/c2_ppo_seed0 --algo ppo

"""Evaluate the best s1-s3 model on the s4 transfer scene (smaller rack + mid-shelf box)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# ── s4 scene override: MUST be set before any env import (scene reads it at import time) ──
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_S4_CONFIG = PROJECT_ROOT / "configs" / "env_config_s4.yaml"
os.environ.setdefault("WAREHOUSE_ENV_CONFIG", str(_S4_CONFIG))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Eval the best s1-s3 model on the s4 transfer scene")
src = parser.add_mutually_exclusive_group(required=True)
src.add_argument("--auto", action="store_true",
                 help="Scan --results for the best run (highest best.json success_rate) and eval it.")
src.add_argument("--run", type=str,
                 help="Path to a single run dir (its best/ holds best_model* + run_config.yaml).")
parser.add_argument("--results", type=str, default="training/results/ablation",
                    help="Root holding the per-config run dirs (used with --auto).")
parser.add_argument("--algo", choices=["ppo", "sac", "dreamer"], default=None,
                    help="Override the algo (else read from the run's run_config.yaml).")
parser.add_argument("--episodes", type=int, default=20, help="Eval episodes on the s4 scene.")
parser.add_argument("--logdir", type=str, default="training/results/s4",
                    help="Where to write s4_metrics.json + the eval CSV.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Project imports (after AppLauncher) ───────────────────────────────
import sys  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT))


def _read_yaml(path: Path) -> dict:
    """Parse a YAML file to a dict ({} if absent/unreadable)."""
    if not path.exists():
        return {}
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _best_success(run_dir: Path) -> float:
    """success_rate recorded in <run_dir>/best/best.json (-inf if missing)."""
    bj = run_dir / "best" / "best.json"
    if not bj.exists():
        return float("-inf")
    try:
        rec = json.loads(bj.read_text(encoding="utf-8"))
    except Exception:
        return float("-inf")
    # best.json nests the eval metrics; accept either flat or {"metrics": {...}}.
    metrics = rec.get("metrics", rec)
    return float(metrics.get("success_rate", float("-inf")))


def _pick_best_run(results_root: Path) -> Path:
    """Run dir with the highest best.json success_rate under results_root."""
    candidates = [d for d in results_root.glob("*") if (d / "best" / "best.json").exists()]
    if not candidates:
        raise SystemExit(f"[eval_s4] no runs with best/best.json under {results_root}")
    best = max(candidates, key=_best_success)
    print(f"[eval_s4] best run = {best.name}  (success_rate={_best_success(best):.3f})")
    return best


def _resolve_algo(run_dir: Path) -> str:
    """Algo for a run: --algo override, else run_config.yaml, else infer from logname prefix."""
    if args_cli.algo:
        return args_cli.algo
    cfg = _read_yaml(run_dir / "best" / "run_config.yaml") or _read_yaml(run_dir / "run_config.yaml")
    algo = cfg.get("algo")
    if algo:
        return str(algo)
    name = run_dir.name
    if "ppo" in name:
        return "ppo"
    if "sac" in name:
        return "sac"
    return "dreamer"


def _sb3_act_fn(run_dir: Path, algo: str, env):
    """Load an SB3 PPO/SAC best_model.zip and return act_fn(obs)->action."""
    model_path = run_dir / "best" / "best_model.zip"
    if not model_path.exists():
        raise SystemExit(f"[eval_s4] missing SB3 checkpoint: {model_path}")
    if algo == "ppo":
        from stable_baselines3 import PPO as Algo
    else:
        from stable_baselines3 import SAC as Algo
    model = Algo.load(str(model_path), env=env)
    print(f"[eval_s4] loaded {algo.upper()} ← {model_path}")
    return lambda obs: model.predict(obs, deterministic=True)[0]


def main() -> None:
    """Build the s4 env, load the best model, run eval, write metrics."""
    # Confirm the override actually points at the s4 scene (guards a stale/missing env var).
    from env import warehouse_scene as ws
    print(f"[eval_s4] scene override = {os.environ['WAREHOUSE_ENV_CONFIG']}")
    print(f"[eval_s4] rack scale = {ws.RACK_SCALE}  shelf_levels = "
          f"{tuple(round(z, 3) for z in ws.RACK_SHELF_LEVELS)}  second_shelf = {ws._SECOND_SHELF}")

    run_dir = _pick_best_run(Path(args_cli.results)) if args_cli.auto else Path(args_cli.run)
    algo = _resolve_algo(run_dir)

    from env.warehouse_env import WarehouseEnvCfg, WarehouseGymEnv
    from experiments.metrics import EvalCsv, evaluate_policy

    cfg = WarehouseEnvCfg()
    cfg.scene.num_envs = 1
    env = WarehouseGymEnv(cfg=cfg)
    try:
        if algo in ("ppo", "sac"):
            from training.env_adapter import SB3WarehouseEnv
            sb3_env = SB3WarehouseEnv(env)
            act = _sb3_act_fn(run_dir, algo, sb3_env)
            eval_env = sb3_env
        else:
            raise SystemExit(
                "[eval_s4] DreamerV3 inference loader not wired here. To eval a DreamerV3 best "
                "model on s4: keep WAREHOUSE_ENV_CONFIG=configs/env_config_s4.yaml set, then run the "
                "DreamerV3 eval path (scripts/train_dreamer.py eval / experiments.nm512_eval) against "
                "this env with the checkpoint at "
                f"{run_dir / 'best'}. The scene override is identical; only the actor load differs.")

        logdir = Path(args_cli.logdir)
        logdir.mkdir(parents=True, exist_ok=True)
        metrics = evaluate_policy(eval_env, act, episodes=args_cli.episodes)
        EvalCsv(logdir).log(0, metrics)
        payload = {"scene": "s4_transfer", "source_run": run_dir.name, "algo": algo,
                   "episodes": args_cli.episodes, "metrics": metrics,
                   "rack_scale": ws.RACK_SCALE, "second_shelf_box": ws._SECOND_SHELF}
        (logdir / "s4_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[eval_s4] s4 success_rate={metrics['success_rate']:.3f} "
              f"mean_return={metrics['mean_return']:.2f} → {logdir / 's4_metrics.json'}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
