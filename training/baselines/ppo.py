# training/baselines/ppo.py
# Person 5 — PPO baseline (model-free, optional; SAC is the priority baseline).
"""PPO baseline wrapper around stable-baselines3 for WarehouseGymEnv."""

from __future__ import annotations

from typing import Any

try:
    from stable_baselines3 import PPO  # type: ignore
    from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore

    _HAS_SB3 = True
except ImportError:
    PPO = None
    DummyVecEnv = None
    _HAS_SB3 = False


DEFAULTS: dict[str, Any] = dict(
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    policy_kwargs=dict(normalize_images=True),
)


def build_ppo(sb3_env, seed: int = 0, tensorboard_log: str | None = None, **overrides):
    """Build an SB3 PPO model on a SB3-compatible env. Returns an unfitted PPO."""
    if not _HAS_SB3:
        raise ImportError(
            "stable-baselines3 not installed. `pip install -r requirements-ml.txt`."
        )
    if DummyVecEnv is not None and not hasattr(sb3_env, "num_envs"):
        sb3_env = DummyVecEnv([lambda: sb3_env])
    kwargs = {**DEFAULTS, **overrides}
    return PPO(
        "MultiInputPolicy",
        sb3_env,
        seed=seed,
        verbose=1,
        tensorboard_log=tensorboard_log,
        **kwargs,
    )
