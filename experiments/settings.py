# experiments/settings.py
# P5 — single editable knob-file loader for the ablation study.
#
# experiments/ablation.yaml holds every value you'd want to tweak between runs (budget,
# eval cadence, CA-SLOPE gains/mode, Visual HER ratio, baseline hyperparameters). This
# module loads that YAML over baked-in defaults so a missing key just falls back. The
# entry scripts read the resolved Settings; nothing here imports Isaac/torch.
#
# Defaults mirror the in-code values so behaviour is identical whether or not a YAML is
# passed. The merge is dict-based and unit-tested (tests/test_experiments.py) without a
# YAML parser via load_settings(overrides=...).

"""Load tunable experiment settings from ablation.yaml over baked-in defaults."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_YAML = Path(__file__).resolve().parent / "ablation.yaml"

# Baked-in defaults (authoritative fallback). Keep in sync with ablation.yaml comments.
_DEFAULTS: dict = {
    "budget": {
        "total_steps": 200_000,
        "seeds": [0, 1, 2],
        "eval_every": 10_000,
        "eval_episodes": 5,
    },
    "ca_slope": {
        "mode": "category",            # category | generic | none
        "gamma": 0.997,
        "category_gains": [1.0, 1.5, 2.0],
        "generic_gain": 1.5,
        "phase_b_offset": 13.0,
    },
    "visual_her": {
        "her_ratio": 0.5,
        "success_reward": 10.0,
    },
    # DreamerV3 uses the vendored paper defaults; only expose the knob most likely tuned.
    "dreamer": {
        "train_ratio": 512,
        "prefill": 2000,
    },
    "sac": {
        "learning_rate": 3e-4, "buffer_size": 200_000, "learning_starts": 5_000,
        "batch_size": 256, "tau": 0.005, "gamma": 0.99,
    },
    "ppo": {
        "learning_rate": 3e-4, "n_steps": 2048, "batch_size": 256, "n_epochs": 10,
        "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0,
    },
}


def _deep_merge(base: dict, over: dict) -> dict:
    """Recursively merge `over` into a copy of `base`."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


@dataclass
class Settings:
    """Resolved experiment settings (defaults merged with the YAML)."""

    budget: dict = field(default_factory=dict)
    ca_slope: dict = field(default_factory=dict)
    visual_her: dict = field(default_factory=dict)
    dreamer: dict = field(default_factory=dict)
    sac: dict = field(default_factory=dict)
    ppo: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        """Build Settings from a fully-merged dict."""
        return cls(budget=d["budget"], ca_slope=d["ca_slope"],
                   visual_her=d["visual_her"], dreamer=d["dreamer"],
                   sac=d["sac"], ppo=d["ppo"])

    def algo_kwargs(self, algo: str) -> dict:
        """Return the SB3 hyperparameter dict for 'sac' or 'ppo'."""
        return dict(self.sac if algo == "sac" else self.ppo)


def _read_yaml(path: Path) -> dict:
    """Parse a YAML file to a dict (pyyaml, then ruamel fallback)."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml
        return yaml.safe_load(text) or {}
    except ImportError:
        import ruamel.yaml as ryaml
        return ryaml.YAML(typ="safe").load(text) or {}


def load_settings(path: str | Path | None = None,
                  overrides: dict | None = None) -> Settings:
    """Load settings: defaults < ablation.yaml (or `path`) < explicit `overrides`.

    Args:
        path:      YAML file to load; defaults to experiments/ablation.yaml if it exists.
                   Pass None and no file -> baked-in defaults only.
        overrides: Final dict merged last (used by tests; skips the parser).

    Returns:
        Settings with all sections populated.
    """
    merged = copy.deepcopy(_DEFAULTS)
    yaml_path = Path(path) if path else DEFAULT_YAML
    if path is not None or yaml_path.exists():
        merged = _deep_merge(merged, _read_yaml(yaml_path))
    if overrides:
        merged = _deep_merge(merged, overrides)
    return Settings.from_dict(merged)
