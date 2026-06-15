# training/seed.py
# Person 5 — reproducibility / seed control.
"""Global seed control for torch, numpy, python-random, CUDA, and env."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = False) -> int:
    """Seed python, numpy, and torch (CPU + CUDA). Returns the seed.

    deterministic=True forces cudnn deterministic + disables benchmark (slower,
    bit-reproducible). Leave False for speed during normal training.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # opt-in: some Isaac/PhysX kernels have no deterministic impl, so only set
        # when the caller explicitly wants bit-reproducibility.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    return seed


def seed_env(env, seed: int) -> None:
    """Seed a WarehouseGymEnv-style env via its reset(seed=...) entry point."""
    try:
        env.reset(seed=seed)
    except TypeError:
        # gymnasium envs that take seed only in reset kwargs already handled above;
        # fall back to a bare reset.
        env.reset()
