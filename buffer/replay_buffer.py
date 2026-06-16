# buffer/replay_buffer.py
# P3 (Jeremy) — Episode-tracking replay buffer + Visual HER hook.
#
# Interface contract (pembagian_tugas.md — all teammates depend on this):
#   buffer.add(obs, action, reward, next_obs, done)
#   buffer.her_relabel(trajectory)          # P3 Visual HER
#   buffer.sample(batch_size) -> Batch
#
# Obs keys (v2 contract, P1 produces):
#   pixels (B,3,64,64), position (B,3), heading (B,2), goal (B,3),
#   goal_id (B,3), ee_pos (B,3), gripper (B,1), holding (B,1), box_pos (B,3)
# Action: (6,) [base_lin, base_ang, ee_dx, ee_dy, ee_dz, gripper] in [-1,1]
#
# Ring buffer with lazy allocation + episode trajectory tracking for HER.
# Everything stored as CPU numpy. Accepts torch or numpy inputs.
# Batch outputs are numpy arrays (convertible to torch by the consumer).

"""Episode-tracking replay buffer with Visual HER support for the warehouse task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


def _to_np(x) -> np.ndarray:
    """Detach torch tensor → numpy; pass numpy through. Strips leading batch dim of 1."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except ImportError:
        pass
    x = np.asarray(x)
    if x.ndim >= 2 and x.shape[0] == 1:
        x = x[0]
    return x


@dataclass
class Batch:
    """Sampled minibatch returned by EpisodeBuffer.sample().

    obs / next_obs: dict[str, np.ndarray] shaped (B, *obs_key_shape)
    action:         np.ndarray (B, 6)
    reward:         np.ndarray (B,)
    done:           np.ndarray (B,) bool
    """

    obs: dict[str, np.ndarray]
    action: np.ndarray
    reward: np.ndarray
    next_obs: dict[str, np.ndarray]
    done: np.ndarray


class EpisodeBuffer:
    """Fixed-capacity ring buffer with episode trajectory tracking for HER.

    Usage:
        buf = EpisodeBuffer(capacity=100_000, her_fn=make_visual_her_fn(zone_pos))

        # per env step:
        buf.add(obs, action, reward, next_obs, done)

        # sample a training batch:
        batch = buf.sample(256)

    When `done=True` is added, the completed episode trajectory is passed to
    `her_fn` and any relabeled transitions are appended to the buffer automatically.

    Obs keys and action shape are inferred lazily from the first add().
    """

    def __init__(
        self,
        capacity: int,
        her_fn: Optional[Callable[[list[dict]], list[dict]]] = None,
        seed: int | None = None,
    ):
        """Create an empty buffer.

        Args:
            capacity: Maximum number of stored transitions (ring overwrites oldest).
            her_fn:   Visual HER relabeling function produced by make_visual_her_fn().
                      Signature: fn(trajectory: list[dict]) -> list[dict].
                      Pass None to disable HER.
            seed:     RNG seed for reproducible sampling.
        """
        self.capacity = int(capacity)
        self._her_fn = her_fn
        self._rng = np.random.default_rng(seed)

        # Ring arrays — allocated lazily on first add().
        self._obs: dict[str, np.ndarray] | None = None
        self._next_obs: dict[str, np.ndarray] | None = None
        self._action: np.ndarray | None = None
        self._reward: np.ndarray | None = None
        self._done: np.ndarray | None = None
        self._ptr = 0
        self._full = False

        # Current-episode trajectory buffer (reset when done=True fires).
        self._episode: list[dict] = []

    # ── public API ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Number of valid transitions currently stored."""
        return self.capacity if self._full else self._ptr

    def add(
        self,
        obs: dict,
        action,
        reward,
        next_obs: dict,
        done,
    ) -> None:
        """Store one or more transitions.

        Batched (B, ...) inputs are unrolled into individual transitions.
        When done=True is stored, the episode trajectory is relabeled via her_fn.

        Args:
            obs:      dict with same keys as P1's obs_v2 contract.
            action:   np.ndarray or torch.Tensor of shape (6,) or (B, 6).
            reward:   scalar or (B,) array.
            next_obs: same keys as obs.
            done:     bool or (B,) bool array.
        """
        obs_np = {k: _to_np(v) for k, v in obs.items()}
        next_obs_np = {k: _to_np(v) for k, v in next_obs.items()}
        action_np = _to_np(action)
        reward_np = _to_np(reward)
        done_np = _to_np(done)

        batched = action_np.ndim > 1
        if not batched:
            self._add_one(
                obs_np, action_np,
                float(reward_np.reshape(-1)[0]),
                next_obs_np,
                bool(done_np.reshape(-1)[0]),
            )
            return

        n = action_np.shape[0]
        for i in range(n):
            self._add_one(
                {k: v[i] for k, v in obs_np.items()},
                action_np[i],
                float(reward_np.reshape(-1)[i]),
                {k: v[i] for k, v in next_obs_np.items()},
                bool(done_np.reshape(-1)[i]),
            )

    def sample(self, batch_size: int) -> Batch:
        """Uniformly sample `batch_size` transitions from the buffer.

        Returns a Batch with numpy arrays of shape (batch_size, ...).
        """
        n = len(self)
        if n == 0:
            raise RuntimeError("EpisodeBuffer is empty — call add() before sample().")
        idx = self._rng.integers(0, n, size=batch_size)
        return Batch(
            obs={k: v[idx] for k, v in self._obs.items()},
            action=self._action[idx],
            reward=self._reward[idx],
            next_obs={k: v[idx] for k, v in self._next_obs.items()},
            done=self._done[idx],
        )

    def her_relabel(self, trajectory: list[dict]) -> None:
        """Apply Visual HER to a completed trajectory and store relabeled transitions.

        Called automatically when done=True fires in add(). Can also be called
        manually with a full episode list.

        Args:
            trajectory: List of step dicts, each:
                {"obs": dict, "action": np.ndarray, "reward": float,
                 "next_obs": dict, "done": bool}
        """
        if self._her_fn is None or not trajectory:
            return
        for t in self._her_fn(trajectory):
            self._add_one(
                t["obs"], np.asarray(t["action"]),
                float(t["reward"]),
                t["next_obs"], bool(t["done"]),
            )

    # ── internal ────────────────────────────────────────────────────────

    def _alloc(self, obs: dict, action: np.ndarray) -> None:
        self._obs = {k: np.zeros((self.capacity, *v.shape), v.dtype) for k, v in obs.items()}
        self._next_obs = {k: np.zeros((self.capacity, *v.shape), v.dtype) for k, v in obs.items()}
        self._action = np.zeros((self.capacity, *action.shape), action.dtype)
        self._reward = np.zeros((self.capacity,), np.float32)
        self._done = np.zeros((self.capacity,), np.bool_)

    def _add_one(
        self,
        obs: dict,
        action: np.ndarray,
        reward: float,
        next_obs: dict,
        done: bool,
    ) -> None:
        if self._obs is None:
            self._alloc(obs, action)

        i = self._ptr
        for k, v in obs.items():
            if k in self._obs:
                self._obs[k][i] = v
                self._next_obs[k][i] = next_obs[k]
        self._action[i] = action
        self._reward[i] = reward
        self._done[i] = done
        self._ptr = (self._ptr + 1) % self.capacity
        self._full = self._full or self._ptr == 0

        # Append to current episode for HER; relabel on done.
        self._episode.append({
            "obs": {k: v.copy() for k, v in obs.items()},
            "action": action.copy(),
            "reward": reward,
            "next_obs": {k: v.copy() for k, v in next_obs.items()},
            "done": done,
        })
        if done:
            self.her_relabel(self._episode)
            self._episode = []
