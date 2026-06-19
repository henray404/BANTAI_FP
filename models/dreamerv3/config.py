# models/dreamerv3/config.py
# Person 2 — build the NM512 dreamerv3-torch config for the warehouse task.
#
# Loads the vendored configs.yaml `defaults`, deep-merges warehouse overrides, and
# returns an argparse.Namespace (NM512 reads attrs like config.encoder["mlp_keys"]).
# Also exposes add_vendor_to_path() so the vendored top-level modules (dreamer,
# models, tools, envs.*) import the way NM512 expects (sys.path, not as a package —
# its internal `import models`/`import tools` would clash with a dotted package).

"""DreamerV3 (NM512) config builder + vendor path setup for the warehouse task."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

VENDOR_DIR = Path(__file__).resolve().parent / "vendor"


def add_vendor_to_path() -> None:
    """Put the vendored dreamerv3-torch dir on sys.path (idempotent)."""
    p = str(VENDOR_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


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


# Warehouse-specific overrides on top of NM512 defaults.
# - obs keys (pickup v2): image (cnn) + position|heading|goal|goal_id|ee_pos|
#   gripper|holding|box_pos (mlp). regex anchored at start (re.match) — "goal" also
#   matches "goal_id" but each key is tested once, so no double-encode.
# - action_repeat=1: the env already decimates 200Hz→10Hz; don't double-repeat.
# - time_limit=1000: pickup episode is 1000 steps (100s @10Hz).
# - prefill/eval kept small so a first end-to-end run is cheap.
_MLP_KEYS = "position|heading|goal|goal_id|ee_pos|gripper|holding|box_pos"
WAREHOUSE_OVERRIDES: dict = {
    "task": "warehouse_pickup",
    "size": [64, 64],
    "envs": 1,
    "action_repeat": 1,
    "time_limit": 1000,
    "prefill": 2000,
    "steps": 200000,
    "eval_every": 10000,
    "eval_episode_num": 5,
    "log_every": 1000,
    "compile": False,            # Windows: torch.compile unsupported anyway
    "encoder": {"mlp_keys": _MLP_KEYS, "cnn_keys": "image"},
    "decoder": {"mlp_keys": _MLP_KEYS, "cnn_keys": "image"},
}


def _deep_merge(base: dict, over: dict) -> dict:
    """Recursively merge `over` into a copy of `base`."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def build_config(extra_overrides: dict | None = None, logdir: str = "training/results/dreamerv3"):
    """Return an argparse.Namespace config for NM512's Dreamer (warehouse task)."""
    import ruamel.yaml as yaml  # vendor dep

    add_vendor_to_path()
    defaults = yaml.YAML(typ="safe").load((VENDOR_DIR / "configs.yaml").read_text())["defaults"]
    cfg = _deep_merge(defaults, WAREHOUSE_OVERRIDES)
    if extra_overrides:
        cfg = _deep_merge(cfg, extra_overrides)
    cfg = _coerce(cfg)
    # vendor defaults ship logdir=null, so setdefault would no-op on the existing
    # None key; assign explicitly (None/'' → fall back to the param).
    cfg["logdir"] = cfg.get("logdir") or logdir
    cfg["traindir"] = None
    cfg["evaldir"] = None
    return argparse.Namespace(**cfg)
