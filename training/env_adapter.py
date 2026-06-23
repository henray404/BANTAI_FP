# training/env_adapter.py
# Person 5 — bridge WarehouseGymEnv → stable-baselines3.
#
# WHY THIS EXISTS:
#   WarehouseGymEnv returns BATCHED torch tensors (num_envs, ...) with a Dict obs
#   whose `pixels` are float[0,1] CHW. SB3 expects a single-env gymnasium.Env with
#   numpy obs, and its CNN feature extractor only treats a key as an IMAGE when the
#   space is uint8. This adapter:
#     - assumes num_envs == 1 and squeezes the batch dim,
#     - converts pixels float[0,1] CHW → uint8[0,255] CHW (so SB3 NatureCNN fires),
#     - converts the rest to numpy float32 vectors,
#     - exposes gymnasium Dict/Box spaces for MultiInputPolicy.
#
# CAVEAT (UNVERIFIED — needs the sim on Blackwell):
#   The underlying Isaac ManagerBasedRLEnv auto-resets a done sub-env on the next
#   step internally. gymnasium/SB3 expect the terminal obs on `done`, then a manual
#   reset(). For num_envs=1 single-episode rollouts this is usually fine, but verify
#   the boundary obs once the sim runs. See bugs_errors/ + docs/project/project_overview.md.

"""Single-env gymnasium adapter exposing WarehouseGymEnv to stable-baselines3."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def _np(x):
    """torch/np → numpy, squeezing a leading batch dim of size 1."""
    try:
        import torch

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except ImportError:
        pass
    x = np.asarray(x)
    if x.ndim >= 1 and x.shape[0] == 1:
        x = x[0]
    return x


class SB3WarehouseEnv(gym.Env):
    """Wrap a num_envs=1 WarehouseGymEnv as a standard single-env gymnasium.Env."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, warehouse_env):
        """Take a built WarehouseGymEnv (must have num_envs == 1)."""
        assert warehouse_env.num_envs == 1, (
            f"SB3WarehouseEnv needs num_envs=1, got {warehouse_env.num_envs}. "
            "SB3 drives a single env; use VecEnv stacking at the SB3 layer instead."
        )
        self._env = warehouse_env
        # IsaacLab's ManagerBasedRLEnv auto-resets a done env INSIDE step() and returns the
        # fresh-episode obs. SB3's VecEnv then calls reset() again on done — a redundant SECOND
        # reset that skips an episode + desyncs _last_obs. We cache the auto-reset obs here and
        # hand it back from reset() so the boundary is a single, consistent reset.
        self._autoreset_obs: dict | None = None
        # obs_v2 + pickup action contract (6,): [base_lin, base_ang, ee_dx, ee_dy, ee_dz, gripper]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)

        IMG = 64
        # obs_v2 (pickup, 2026-06-08): pixels + 8 low-dim keys. goal_emb removed → goal_id one-hot.
        self._vec_keys = {
            "position": 3, "heading": 2, "goal": 3, "goal_id": 3,
            "ee_pos": 3, "gripper": 1, "holding": 1, "box_pos": 3,
        }
        spaces_d = {
            k: spaces.Box(-np.inf, np.inf, shape=(d,), dtype=np.float32)
            for k, d in self._vec_keys.items()
        }
        # uint8 image → SB3 treats it as a CNN input and auto-normalizes.
        spaces_d["pixels"] = spaces.Box(0, 255, shape=(3, IMG, IMG), dtype=np.uint8)
        self.observation_space = spaces.Dict(spaces_d)

    def _convert(self, obs: dict) -> dict:
        """Squeeze batch, cast pixels → uint8, low-dim keys → float32 (obs_v2)."""
        px = _np(obs["pixels"])
        if px.dtype != np.uint8:
            px = np.clip(px, 0.0, 1.0) * 255.0
            px = px.astype(np.uint8)
        out = {"pixels": px}
        for k in self._vec_keys:
            out[k] = _np(obs[k]).astype(np.float32).reshape(-1)
        return out

    def reset(self, *, seed: int | None = None, options=None):
        """Reset underlying env; return (obs, info).

        If the underlying env already auto-reset on the previous done step, hand back that
        fresh-episode obs instead of resetting a SECOND time (both are valid uniformly-random
        episode starts; the double reset just wasted a sim reset and desynced _last_obs).
        """
        super().reset(seed=seed)
        if self._autoreset_obs is not None:
            obs = self._autoreset_obs
            self._autoreset_obs = None
            return obs, {}
        obs, info = self._env.reset(seed=seed)
        return self._convert(obs), dict(info) if isinstance(info, dict) else {}

    def step(self, action):
        """Apply action; return (obs, reward, terminated, truncated, info)."""
        action = np.asarray(action, dtype=np.float32).reshape(6)
        obs, reward, terminated, truncated, info = self._env.step(action)
        r = float(_np(reward).reshape(-1)[0])
        term = bool(_np(terminated).reshape(-1)[0])
        trunc = bool(_np(truncated).reshape(-1)[0])
        conv = self._convert(obs)
        # On done, `conv` is already the auto-reset (next-episode) obs — cache it so the VecEnv's
        # follow-up reset() returns it rather than triggering a redundant second reset.
        self._autoreset_obs = conv if (term or trunc) else None
        return conv, r, term, trunc, {}

    def render(self):
        """Return env-0 camera RGB (uint8 HWC)."""
        return self._env.render()

    def close(self):
        """Close underlying env."""
        self._env.close()
