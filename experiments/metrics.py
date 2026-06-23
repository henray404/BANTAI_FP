# experiments/metrics.py
# P5 — shared evaluation + per-run metric logging for all 6 configurations.
#
# Metrics (spec "Metrik evaluasi"):
#   (1) success rate  : fraction of eval episodes that delivered the correct-category
#                       box to the correct color zone and released it (pickup_delivered).
#   (2) sample eff.   : env steps to reach a success-rate threshold (computed offline in
#                       analyze.py from the logged eval rows).
#   (3) episode length: mean steps to episode end during eval.
#
# evaluate_policy() works for ANY stack (Dreamer / SAC / PPO): it only needs the
# warehouse gym env (possibly wrapped) + a callable act_fn(obs)->action. EvalCsv writes
# one row per periodic eval; analyze.py consumes the CSVs across seeds.

"""Evaluation loop (success rate / length / return) + per-run CSV logging."""

from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path
from typing import Callable

import numpy as np

CSV_HEADER = ("step", "success_rate", "success_std", "mean_length", "mean_return")


def _rl_env(env):
    """Unwrap gym/SB3 wrappers down to the WarehouseRLEnv that holds the state buffers.

    Descends through BOTH the gym `.env` chain and the project `._env` nesting until it
    reaches the object carrying the reward buffers (WarehouseRLEnv has `box_pos` + `scene`).
    The old single `e._env` stopped one level too shallow on the SB3 path
    (SB3WarehouseEnv._env == WarehouseGymEnv, NOT WarehouseRLEnv) → AttributeError inside
    pickup_delivered / capture_init_state, which crashed every PPO/SAC eval.
    """
    e = env
    seen: set[int] = set()
    while id(e) not in seen:
        seen.add(id(e))
        if hasattr(e, "box_pos") and hasattr(e, "scene"):
            return e
        nxt = getattr(e, "_env", None)
        if nxt is None:
            nxt = getattr(e, "env", None)
        if nxt is None or nxt is e:
            break
        e = nxt
    return getattr(e, "_env", e)  # fallback: preserve old behaviour


def episode_success(env) -> bool:
    """True if the (single) env currently has the correct box delivered in its zone.

    NOTE: reads the LIVE buffers — only valid BEFORE the env auto-resets. evaluate_policy no
    longer uses this for scoring (it latches `_delivered_this_step` during the rollout); kept
    for callers/tests that hold a pre-reset env.
    """
    from env.reward_pickup import pickup_delivered

    return bool(pickup_delivered(_rl_env(env)).reshape(-1)[0])


def _delivered_this_step(rl, info) -> bool:
    """True if the delivery/success condition fired on the step just taken.

    Reads the env info flag first (the dreamer path surfaces 'deliver_event'); falls back to
    the termination manager's 'success' term, which still reflects the terminating step even
    after IsaacLab's in-step auto-reset (the same read the WarehouseGymEnv DIAG uses). This
    avoids the old bug where pickup_delivered() ran on the already-reset state → always 0.
    """
    if isinstance(info, dict):
        for k in ("success", "deliver_event"):
            if k in info:
                try:
                    return bool(np.asarray(_to_np(info[k])).reshape(-1)[0])
                except Exception:
                    return bool(info[k])
    tm = getattr(rl, "termination_manager", None)
    if tm is not None and "success" in getattr(tm, "active_terms", []):
        try:
            return bool(tm.get_term("success")[0].item())
        except Exception:
            return False
    return False


def evaluate_policy(env, act_fn: Callable[[dict], np.ndarray],
                    episodes: int = 5, recorder=None) -> dict[str, float]:
    """Run `episodes` deterministic eval episodes; return success/length/return means.

    Args:
        env:      Warehouse gym env (num_envs=1), optionally wrapped.
        act_fn:   Maps an obs dict to a (6,) numpy action (use the greedy/mean policy).
        episodes: Number of eval episodes.
        recorder: Optional TrajectoryRecorder — saves the best episode's action trace
                  + scene snapshot for GUI replay.

    Returns:
        {"success_rate": ..., "success_std": ..., "mean_length": ..., "mean_return": ...}
    """
    successes, lengths, returns = [], [], []
    rl = _rl_env(env)  # stable across resets; unwrap once
    for _ in range(episodes):
        obs, _ = env.reset()
        if recorder is not None:
            from env.scene_snapshot import capture_init_state
            recorder.begin(capture_init_state(rl))
        done, ep_len, ep_ret, delivered = False, 0, 0.0, False
        while not done:
            action = act_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            r = float(np.asarray(_to_np(reward)).reshape(-1)[0])
            ep_ret += r
            ep_len += 1
            # Latch success on the terminating step: the env auto-resets its buffers inside
            # step(), so the old post-loop pickup_delivered() check always read 0.
            delivered = delivered or _delivered_this_step(rl, info)
            if recorder is not None:
                from env.scene_snapshot import read_replay_state
                robot_xyz, ee_xyz, holding = read_replay_state(rl)
                recorder.step(ep_len, action, robot_xyz, ee_xyz, holding, r)
            term = bool(np.asarray(_to_np(terminated)).reshape(-1)[0])
            trunc = bool(np.asarray(_to_np(truncated)).reshape(-1)[0])
            done = term or trunc
            if term:
                break
        succ = 1.0 if delivered else 0.0
        successes.append(succ)
        lengths.append(ep_len)
        returns.append(ep_ret)
        if recorder is not None:
            recorder.end(succ, ep_ret)
    return {
        "success_rate": float(np.mean(successes)),
        "success_std": float(np.std(successes)),
        "mean_length": float(np.mean(lengths)),
        "mean_return": float(np.mean(returns)),
    }


def _to_np(x):
    """torch/np scalar/tensor -> numpy (no torch import unless needed)."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


class EvalCsv:
    """Append-only writer of periodic eval rows to <logdir>/eval_metrics.csv."""

    def __init__(self, logdir: str | Path, filename: str = "eval_metrics.csv"):
        """Create (with header) the eval CSV under `logdir`."""
        self.path = Path(logdir) / filename
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="") as f:
                csv.writer(f).writerow(CSV_HEADER)

    def log(self, step: int, metrics: dict[str, float]) -> None:
        """Append one eval row for env-step `step`."""
        with self.path.open("a", newline="") as f:
            csv.writer(f).writerow([
                int(step),
                metrics["success_rate"],
                metrics.get("success_std", 0.0),
                metrics["mean_length"],
                metrics["mean_return"],
            ])


def write_run_config(logdir: str | Path, payload: dict) -> Path:
    """Snapshot the exact run configuration to <logdir>/run_config.yaml (reproducibility).

    Records resolved settings + flags (algo, seed, ca_slope, visual_her, steps, ...) so a
    run — especially the best one — can be reproduced from disk. YAML if available, else JSON.

    Returns the written path.
    """
    logdir = Path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    out = logdir / "run_config.yaml"
    try:
        import yaml
        out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    except ImportError:
        out = logdir / "run_config.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


class BestModelTracker:
    """Persist the best-so-far checkpoint + its metrics + config so it can be re-run.

    On each eval, call update(step, metrics, checkpoint_src). When success_rate improves
    (ties broken by mean_return), it writes <logdir>/best/best.json, copies the run config,
    and copies the current checkpoint to <logdir>/best/best_model<ext>.
    """

    def __init__(self, logdir: str | Path):
        """Create <logdir>/best/ and seed the best score to -inf."""
        self.dir = Path(logdir) / "best"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._best_sr = float("-inf")
        self._best_ret = float("-inf")

    def update(self, step: int, metrics: dict[str, float],
               checkpoint_src: str | Path | None = None) -> bool:
        """Save a new best snapshot if this eval beats the previous best. Returns True if saved."""
        sr = metrics["success_rate"]
        ret = metrics.get("mean_return", 0.0)
        if (sr < self._best_sr) or (sr == self._best_sr and ret <= self._best_ret):
            return False
        self._best_sr, self._best_ret = sr, ret

        record = {"step": int(step), "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                  **{k: float(v) for k, v in metrics.items()}}
        (self.dir / "best.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

        run_cfg = self.dir.parent / "run_config.yaml"
        if run_cfg.exists():
            shutil.copy2(run_cfg, self.dir / "run_config.yaml")

        if checkpoint_src is not None and Path(checkpoint_src).exists():
            ext = Path(checkpoint_src).suffix or ".pt"
            shutil.copy2(checkpoint_src, self.dir / f"best_model{ext}")
        return True
