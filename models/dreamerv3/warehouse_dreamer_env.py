# models/dreamerv3/warehouse_dreamer_env.py
# Person 2 — NM512-suite-style env wrapping WarehouseGymEnv for DreamerV3.
#
# Mirrors models/dreamerv3/vendor/envs/dmc.py:DeepMindControl so the vendored NM512
# wrapper stack (TimeLimit, SelectAction, UUID) + tools.simulate work unchanged.
# Uses OLD gym (gym==0.22, vendor dep) for spaces — NOT gymnasium — because the
# vendored wrappers subclass old gym.Wrapper.
#
# NO AppLauncher here. The training entry (scripts/train_dreamer.py) owns it and
# passes a built WarehouseGymEnv (num_envs=1) into the factory.
#
# UNVERIFIED: full env needs `pixels` → blocked by the Blackwell camera SDP issue on
# the reference RTX 5050 (docs/project/project_overview.md). Run on a working sim.

"""NM512 dreamerv3-torch suite env wrapping WarehouseGymEnv (num_envs=1)."""

from __future__ import annotations

import gym  # vendor dep: old gym 0.22 (NOT gymnasium)
import numpy as np

from .obs_adapter import VECTOR_KEYS, pixels_to_image, warehouse_obs_to_dreamer


class WarehouseDreamer:
    """Single-env warehouse task in NM512's expected env API.

    obs dict per step: image (64,64,3 uint8) + position/goal/goal_emb/heading floats
    + is_first + is_terminal. reset()→obs, step(a)→(obs, reward, done, info).
    """

    metadata = {}

    def __init__(self, warehouse_env, size=(64, 64)):
        """Take a built WarehouseGymEnv with num_envs == 1."""
        assert warehouse_env.num_envs == 1, "WarehouseDreamer needs num_envs=1."
        self._env = warehouse_env
        self._size = size
        self.reward_range = [-np.inf, np.inf]
        self._last_image = np.zeros(size + (3,), np.uint8)

    @property
    def observation_space(self):
        """Dict space: image (uint8) + flat float vectors (mlp_keys)."""
        dims = {"position": 3, "goal": 3, "goal_emb": 512, "heading": 2}
        spaces = {
            k: gym.spaces.Box(-np.inf, np.inf, (dims[k],), dtype=np.float32)
            for k in VECTOR_KEYS
        }
        spaces["image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        """[linear, angular] in [-1, 1]."""
        return gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

    def _obs(self, raw: dict, is_first: bool, is_terminal: bool) -> dict:
        """Build the NM512 obs dict from a warehouse obs dict."""
        obs = warehouse_obs_to_dreamer(raw)
        self._last_image = obs["image"]
        obs["is_first"] = is_first
        obs["is_terminal"] = is_terminal
        return obs

    def reset(self):
        """Reset; return the initial obs dict (is_first=True)."""
        raw, _ = self._env.reset()
        return self._obs(raw, is_first=True, is_terminal=False)

    def step(self, action):
        """Apply [linear, angular]; return (obs, reward, done, info)."""
        action = np.asarray(action, dtype=np.float32).reshape(2)
        raw, reward, terminated, truncated, _ = self._env.step(action)
        r = float(np.asarray(_to_np(reward)).reshape(-1)[0])
        term = bool(np.asarray(_to_np(terminated)).reshape(-1)[0])
        trunc = bool(np.asarray(_to_np(truncated)).reshape(-1)[0])
        done = term or trunc
        # is_terminal = true terminal (goal/out-of-bounds), NOT a time-limit truncation.
        obs = self._obs(raw, is_first=False, is_terminal=term)
        return obs, r, done, {"discount": np.array(0.0 if term else 1.0, np.float32)}

    def render(self, *args, **kwargs):
        """Return the last camera frame (HWC uint8)."""
        return self._last_image

    def close(self):
        """Close underlying env."""
        self._env.close()


def _to_np(x):
    """torch/np scalar/tensor → numpy."""
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def make_warehouse_dreamer(warehouse_env, config):
    """Factory mirroring vendor make_env tail: wrap with NM512's wrapper stack.

    Returns an env ready for tools.simulate / Dreamer. `config` provides time_limit.
    """
    from models.dreamerv3.config import add_vendor_to_path

    add_vendor_to_path()
    import envs.wrappers as wrappers  # vendored top-level module (on sys.path)

    env = WarehouseDreamer(warehouse_env, size=tuple(config.size))
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    return env
