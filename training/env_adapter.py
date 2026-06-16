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
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        IMG = 64
        EMB = 512
        self.observation_space = spaces.Dict(
            {
                # uint8 image → SB3 treats it as a CNN input and auto-normalizes.
                "pixels":   spaces.Box(0, 255, shape=(3, IMG, IMG), dtype=np.uint8),
                "position": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "goal":     spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "goal_emb": spaces.Box(-np.inf, np.inf, shape=(EMB,), dtype=np.float32),
                "heading":  spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
            }
        )

    def _convert(self, obs: dict) -> dict:
        """Squeeze batch, cast pixels → uint8, others → float32."""
        px = _np(obs["pixels"])
        if px.dtype != np.uint8:
            px = np.clip(px, 0.0, 1.0) * 255.0
            px = px.astype(np.uint8)
        return {
            "pixels":   px,
            "position": _np(obs["position"]).astype(np.float32),
            "goal":     _np(obs["goal"]).astype(np.float32),
            "goal_emb": _np(obs["goal_emb"]).astype(np.float32),
            "heading":  _np(obs["heading"]).astype(np.float32),
        }

    def reset(self, *, seed: int | None = None, options=None):
        """Reset underlying env; return (obs, info)."""
        super().reset(seed=seed)
        obs, info = self._env.reset(seed=seed)
        return self._convert(obs), dict(info) if isinstance(info, dict) else {}

    def step(self, action):
        """Apply action; return (obs, reward, terminated, truncated, info)."""
        action = np.asarray(action, dtype=np.float32).reshape(2)
        obs, reward, terminated, truncated, info = self._env.step(action)
        r = float(_np(reward).reshape(-1)[0])
        term = bool(_np(terminated).reshape(-1)[0])
        trunc = bool(_np(truncated).reshape(-1)[0])
        return self._convert(obs), r, term, trunc, {}

    def render(self):
        """Return env-0 camera RGB (uint8 HWC)."""
        return self._env.render()

    def close(self):
        """Close underlying env."""
        self._env.close()
