# scripts/train_offline.py — OFFLINE DreamerV3 training (NO Isaac Sim, Colab-ready).
#
# Why this exists: Isaac Sim does not run on Google Colab (no Omniverse, fixed
# driver, no offscreen camera). So we split compute:
#   1. Collect rollouts LOCALLY with the sim  -> scripts/collect_offline.py
#   2. Upload the episode dir to Colab, train the nets HERE on an A100.
#
# This script imports NOTHING from the project `models`/`env`/`policy` packages
# (those drag Isaac + P3 deps). It only needs the vendored NM512 code
# (models/dreamerv3/vendor) + torch + ruamel.yaml. `import models`/`import tools`
# resolve to the VENDOR dir because only that dir is added to sys.path.
#
# The checkpoint is Dreamer-compatible: submodule names "_wm.*"/"_task_behavior.*"
# /"_expl_behavior.*" (greedy) match scripts/train_dreamer.py's latest.pt, so the
# online eval can load it later.
#
# Usage (Colab A100 or local):
#   python scripts/train_offline.py --data training/data/episodes --steps 100000
#   python scripts/train_offline.py --data <dir> --batch_size 32 --device cuda
#   python scripts/train_offline.py --self_check          # no data/GPU needed

"""Offline DreamerV3 (NM512) training from a pre-collected replay dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = PROJECT_ROOT / "models" / "dreamerv3" / "vendor"
sys.path.insert(0, str(VENDOR_DIR))  # ONLY the vendor dir → bare `import models` = NM512

# Keys stored next to obs in each episode that are NOT model observations.
_NON_OBS = {"action", "reward", "discount", "is_first", "is_last",
            "is_terminal", "logprob", "id", "uuid"}

# Warehouse overrides on NM512 defaults — kept in sync with models/dreamerv3/config.py
# (duplicated on purpose so this script has ZERO project-package imports for Colab).
_MLP_KEYS = "position|heading|goal|goal_id|ee_pos|gripper|holding|box_pos"
_WAREHOUSE_OVERRIDES: dict = {
    "task": "warehouse_pickup",
    "size": [64, 64],
    "envs": 1,
    "action_repeat": 1,
    "time_limit": 1000,
    "compile": False,
    "encoder": {"mlp_keys": _MLP_KEYS, "cnn_keys": "image"},
    "decoder": {"mlp_keys": _MLP_KEYS, "cnn_keys": "image"},
}


# ── config (inline, no project import) ─────────────────────────────────────────

def _coerce(v):
    """Cast yaml scalar strings like '1e6' / '3e-4' to numbers where possible."""
    if isinstance(v, str):
        try:
            f = float(v)
            return int(f) if f.is_integer() and "e" not in v.lower() and "." not in v else f
        except ValueError:
            return v
    if isinstance(v, dict):
        return {k: _coerce(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_coerce(x) for x in v]
    return v


def _deep_merge(base: dict, over: dict) -> dict:
    """Recursively merge `over` into a copy of `base`."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def build_config(overrides: dict, logdir: str, device: str) -> argparse.Namespace:
    """Build the NM512 warehouse config Namespace (vendor defaults + overrides)."""
    import ruamel.yaml as yaml

    defaults = yaml.YAML(typ="safe").load((VENDOR_DIR / "configs.yaml").read_text())["defaults"]
    cfg = _deep_merge(defaults, _WAREHOUSE_OVERRIDES)
    cfg = _deep_merge(cfg, overrides)
    cfg = _coerce(cfg)
    cfg["logdir"] = logdir
    cfg["traindir"] = None
    cfg["evaldir"] = None
    cfg["device"] = device
    cfg["num_actions"] = 6
    return argparse.Namespace(**cfg)


# ── obs/act space shims (built from data — no gym dependency) ──────────────────

class _Box:
    """gym.spaces.Box stand-in; the WM only reads .shape / .dtype."""

    def __init__(self, shape, dtype) -> None:
        self.shape = tuple(shape)
        self.dtype = dtype


class _Dict:
    """gym.spaces.Dict stand-in exposing .spaces (WM reads each v.shape)."""

    def __init__(self, spaces: dict) -> None:
        self.spaces = spaces


def spaces_from_episode(ep: dict, mlp_keys: str, cnn_keys: str) -> tuple[_Dict, _Box]:
    """Derive obs Dict + action Box from one loaded episode's arrays (drops time dim)."""
    import re

    obs = {}
    for k, v in ep.items():
        if k in _NON_OBS or not hasattr(v, "shape") or v.dtype.kind not in "fui":
            continue
        if re.match(cnn_keys, k) or re.match(mlp_keys, k):
            obs[k] = _Box(v.shape[1:], v.dtype)
    if not obs:
        raise ValueError(f"No obs keys matched encoder regex in episode keys: {list(ep)}")
    return _Dict(obs), _Box(ep["action"].shape[1:], ep["action"].dtype)


# ── offline agent (Dreamer-compatible state_dict, greedy exploration) ──────────

class OfflineAgent(nn.Module):
    """WM + actor-critic trained purely in imagination from a fixed dataset."""

    def __init__(self, obs_space, act_space, config) -> None:
        """Build the world model and imagination behavior (vendor NM512 modules)."""
        super().__init__()
        import models  # vendor/models.py (only VENDOR_DIR on sys.path)

        self._wm = models.WorldModel(obs_space, act_space, 0, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)
        self._expl_behavior = self._task_behavior  # greedy: shared → matches Dreamer

    def train_step(self, data) -> dict:
        """One WM update + one actor-critic imagination update; return metrics."""
        post, _, mets = self._wm._train(data)
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)).mode()
        mets.update(self._task_behavior._train(post, reward)[-1])
        return mets


# ── checkpoint ─────────────────────────────────────────────────────────────────

def _save(agent: OfflineAgent, tools, path: Path, step: int) -> None:
    """Write a Dreamer-format checkpoint (agent + optim state + step)."""
    torch.save({
        "agent_state_dict": agent.state_dict(),
        "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        "step": step,
    }, path)


# ── main ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    """CLI for the offline trainer."""
    p = argparse.ArgumentParser(description="Offline DreamerV3 training (no sim)")
    p.add_argument("--data", type=str, help="Dir of .npz episodes (from collect_offline.py)")
    p.add_argument("--logdir", type=str, default="training/results/offline")
    p.add_argument("--steps", type=int, default=100_000, help="Gradient steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch_size", type=int, default=None, help="Override NM512 batch_size")
    p.add_argument("--batch_length", type=int, default=None, help="Override NM512 batch_length")
    p.add_argument("--log_every", type=int, default=1000)
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--video_pred", action="store_true", help="Log WM reconstruction video")
    p.add_argument("--self_check", action="store_true", help="Run a no-data sanity check and exit")
    return p.parse_args()


def _self_check() -> None:
    """Build spaces from a synthetic episode and assert shapes — no torch/GPU/data."""
    T = 8
    ep = {
        "image": np.zeros((T, 64, 64, 3), np.uint8),
        "position": np.zeros((T, 3), np.float32),
        "goal_id": np.zeros((T, 3), np.float32),
        "action": np.zeros((T, 6), np.float32),
        "reward": np.zeros((T,), np.float32),
        "is_first": np.zeros((T,), bool),
        "id": np.array(["x"] * T),  # string bookkeeping → must be ignored
    }
    obs, act = spaces_from_episode(ep, _MLP_KEYS, "image")
    assert obs.spaces["image"].shape == (64, 64, 3), obs.spaces["image"].shape
    assert obs.spaces["position"].shape == (3,)
    assert "id" not in obs.spaces and "reward" not in obs.spaces
    assert act.shape == (6,), act.shape
    print("[self_check] OK — spaces built, bookkeeping keys excluded.")


def main() -> None:
    """Load the offline dataset, build the agent, and run the training loop."""
    args = _parse_args()
    if args.self_check:
        _self_check()
        return
    if not args.data:
        raise SystemExit("--data is required (dir of .npz episodes). Or use --self_check.")

    overrides: dict = {"seed": args.seed, "steps": args.steps, "log_every": args.log_every}
    if args.batch_size:
        overrides["batch_size"] = args.batch_size
    if args.batch_length:
        overrides["batch_length"] = args.batch_length
    config = build_config(overrides, logdir=args.logdir, device=args.device)

    import tools  # vendor/tools.py

    logdir = Path(config.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    tools.set_seed_everywhere(args.seed)

    episodes = tools.load_episodes(Path(args.data), limit=config.dataset_size)
    if not episodes:
        raise SystemExit(f"No .npz episodes found in {args.data}")
    sample = next(iter(episodes.values()))
    obs_space, act_space = spaces_from_episode(
        sample, config.encoder["mlp_keys"], config.encoder["cnn_keys"])
    print(f"[data] {len(episodes)} episodes | obs keys: {list(obs_space.spaces)}", flush=True)

    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)

    logger = tools.Logger(logdir, 0)
    agent = OfflineAgent(obs_space, act_space, config).to(config.device)
    agent.requires_grad_(requires_grad=False)

    start = 0
    ckpt = logdir / "latest.pt"
    if ckpt.exists():
        sd = torch.load(ckpt, map_location=config.device)
        agent.load_state_dict(sd["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, sd["optims_state_dict"])
        start = sd.get("step", 0)
        print(f"[resume] {ckpt} @ step {start}", flush=True)

    print(f"[train] {start} → {args.steps} steps on {config.device}", flush=True)
    metrics: dict = {}
    for step in range(start, args.steps):
        for k, v in agent.train_step(next(dataset)).items():
            metrics.setdefault(k, []).append(v)
        if (step + 1) % args.log_every == 0:
            logger.step = step + 1
            for k, vals in metrics.items():
                logger.scalar(k, float(np.mean(vals)))
            metrics = {}
            if args.video_pred:
                openl = agent._wm.video_pred(next(dataset))
                logger.video("train_openl", openl.detach().cpu().numpy())
            logger.write(fps=True)
        if (step + 1) % args.save_every == 0:
            _save(agent, tools, ckpt, step + 1)
            print(f"[ckpt] {ckpt} @ step {step + 1}", flush=True)

    _save(agent, tools, ckpt, args.steps)
    print(f"[done] checkpoint → {ckpt}", flush=True)


if __name__ == "__main__":
    main()
