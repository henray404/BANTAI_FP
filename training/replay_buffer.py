# training/replay_buffer.py
# Person 5 — experience replay (shared: SAC + Visual HER).
#
# Stores Dict observations matching the WarehouseGymEnv interface contract:
#   obs = {pixels, position, goal, goal_emb, heading}
# Everything is stored as CPU numpy in pre-allocated ring arrays. Accepts torch
# or numpy inputs. HER relabeling hook lives here (Person 4 fills relabel_fn).

"""Transition-level replay buffer with a Dict-obs layout and an HER relabel hook."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def _to_np(x) -> np.ndarray:
    """Detach torch → numpy; pass numpy through. Drops a leading batch dim of 1."""
    try:
        import torch

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except ImportError:
        pass
    x = np.asarray(x)
    return x


@dataclass
class Batch:
    """Sampled minibatch. obs/next_obs are dict[str, np.ndarray] of shape (B, ...)."""

    obs: dict[str, np.ndarray]
    action: np.ndarray
    reward: np.ndarray
    next_obs: dict[str, np.ndarray]
    done: np.ndarray


class ReplayBuffer:
    """Fixed-capacity ring buffer over Dict-obs transitions.

    Interface (matches training/README):
        buffer.add(obs, action, reward, next_obs, done)
        batch = buffer.sample(batch_size)            -> Batch
        buffer.her_relabel(...)                      -> Visual HER (Person 4)

    obs keys + shapes are inferred from the first add(). Inputs may be torch or
    numpy, single-env (no batch dim) or batched (B, ...) — batched adds are split
    into individual transitions.
    """

    def __init__(self, capacity: int, seed: int | None = None):
        """Allocate lazily on first add; `capacity` = max stored transitions."""
        self.capacity = int(capacity)
        self._rng = np.random.default_rng(seed)
        self._obs: dict[str, np.ndarray] | None = None
        self._next_obs: dict[str, np.ndarray] | None = None
        self._action: np.ndarray | None = None
        self._reward: np.ndarray | None = None
        self._done: np.ndarray | None = None
        self._ptr = 0
        self._full = False

    def __len__(self) -> int:
        """Number of valid transitions currently stored."""
        return self.capacity if self._full else self._ptr

    def _alloc(self, obs: dict, action: np.ndarray) -> None:
        """Pre-allocate ring arrays from the first transition's shapes/dtypes."""
        self._obs = {k: np.zeros((self.capacity, *v.shape), v.dtype) for k, v in obs.items()}
        self._next_obs = {k: np.zeros((self.capacity, *v.shape), v.dtype) for k, v in obs.items()}
        self._action = np.zeros((self.capacity, *action.shape), action.dtype)
        self._reward = np.zeros((self.capacity,), np.float32)
        self._done = np.zeros((self.capacity,), np.bool_)

    def add(self, obs: dict, action, reward, next_obs: dict, done) -> None:
        """Store one or more transitions. Batched (B,...) inputs are unrolled."""
        obs = {k: _to_np(v) for k, v in obs.items()}
        next_obs = {k: _to_np(v) for k, v in next_obs.items()}
        action = _to_np(action)
        reward = _to_np(reward)
        done = _to_np(done)

        # Detect batch: action with one extra leading dim relative to a single (2,) action.
        batched = action.ndim > 1
        if not batched:
            self._add_one(obs, action, float(np.asarray(reward).reshape(-1)[0]),
                          next_obs, bool(np.asarray(done).reshape(-1)[0]))
            return
        n = action.shape[0]
        for i in range(n):
            self._add_one(
                {k: v[i] for k, v in obs.items()},
                action[i],
                float(reward.reshape(-1)[i]),
                {k: v[i] for k, v in next_obs.items()},
                bool(done.reshape(-1)[i]),
            )

    def _add_one(self, obs, action, reward, next_obs, done) -> None:
        """Write a single transition at the ring pointer."""
        if self._obs is None:
            self._alloc(obs, action)
        i = self._ptr
        for k, v in obs.items():
            self._obs[k][i] = v
            self._next_obs[k][i] = next_obs[k]
        self._action[i] = action
        self._reward[i] = reward
        self._done[i] = done
        self._ptr = (self._ptr + 1) % self.capacity
        self._full = self._full or self._ptr == 0

    def sample(self, batch_size: int) -> Batch:
        """Uniformly sample `batch_size` transitions."""
        n = len(self)
        if n == 0:
            raise RuntimeError("ReplayBuffer empty — add transitions before sampling.")
        idx = self._rng.integers(0, n, size=batch_size)
        return Batch(
            obs={k: v[idx] for k, v in self._obs.items()},
            action=self._action[idx],
            reward=self._reward[idx],
            next_obs={k: v[idx] for k, v in self._next_obs.items()},
            done=self._done[idx],
        )

    # ── Visual HER hook (Person 4) ───────────────────────────────────────
    def her_relabel(self, trajectory: list[dict], relabel_fn: Callable) -> None:
        """Append HER-relabeled copies of a finished episode.

        `trajectory` = list of per-step dicts {obs, action, reward, next_obs, done}.
        `relabel_fn(trajectory) -> list[transition]` is supplied by Visual HER
        (perception/language/visual_her.py): it rewrites goal/goal_emb/reward for
        the category the robot actually approached, then we store the result.
        """
        for t in relabel_fn(trajectory):
            self.add(t["obs"], t["action"], t["reward"], t["next_obs"], t["done"])
