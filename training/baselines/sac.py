# training/baselines/sac.py
# Person 5 — SAC baseline (model-free) for RQ1 comparison vs DreamerV3.
#
# Uses stable-baselines3 SAC + MultiInputPolicy (Dict obs: image + vectors).
# Import-guarded so the module imports even without SB3 installed; build_sac()
# raises a clear message if it's missing. See requirements-ml.txt.

"""SAC baseline wrapper around stable-baselines3 for WarehouseGymEnv."""

from __future__ import annotations

from typing import Any

try:
    from stable_baselines3 import SAC  # type: ignore
    from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore

    _HAS_SB3 = True
except ImportError:
    SAC = None
    DummyVecEnv = None
    _HAS_SB3 = False


# SAC defaults tuned for a Dict (image+vector) obs warehouse nav task. Override via cfg.
DEFAULTS: dict[str, Any] = dict(
    learning_rate=3e-4,
    buffer_size=200_000,      # 8GB-conscious; raise if RAM allows
    learning_starts=5_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    policy_kwargs=dict(normalize_images=True),  # uint8 pixels → /255 in CNN
)


def _require_sb3() -> None:
    if not _HAS_SB3:
        raise ImportError(
            "stable-baselines3 not installed. `pip install -r requirements-ml.txt` "
            "(installs sb3). Keep it OUT of the pinned isaaclab torch env if it tries "
            "to downgrade torch 2.7.0+cu128 — use --no-deps or a separate env."
        )


def build_sac(sb3_env, seed: int = 0, tensorboard_log: str | None = None, **overrides):
    """Build an SB3 SAC model on a SB3-compatible env (see training/env_adapter).

    `sb3_env` is an SB3WarehouseEnv (or a VecEnv of one). Returns an unfitted SAC.
    """
    _require_sb3()
    if DummyVecEnv is not None and not hasattr(sb3_env, "num_envs"):
        sb3_env = DummyVecEnv([lambda: sb3_env])
    kwargs = {**DEFAULTS, **overrides}
    return SAC(
        "MultiInputPolicy",
        sb3_env,
        seed=seed,
        verbose=1,
        tensorboard_log=tensorboard_log,
        **kwargs,
    )
