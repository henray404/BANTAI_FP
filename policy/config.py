# policy/config.py
# P3 (Jeremy) — Hyperparameter config for actor-critic + training loop.

"""P3 hyperparameter config dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class P3Config:
    """All P3 hyperparameters in one place.

    Defaults tuned for DreamerV3-small on the warehouse pickup task.
    Override per experiment; pass to P3Trainer.

    RSSM dims (must match P2's world model config):
        feat_dim = dyn_deter + dyn_stoch * dyn_discrete
                 = 512      + 32         * 32
                 = 1536
    """

    # ── RSSM interface (must match P2) ──────────────────────────────────
    feat_dim: int = 1536        # DreamerV3-small: 512 + 32*32 = 1536

    # ── Action ──────────────────────────────────────────────────────────
    action_dim: int = 6         # [base_lin, base_ang, ee_dx, ee_dy, ee_dz, gripper]

    # ── Actor ───────────────────────────────────────────────────────────
    actor_hidden: list[int] = field(default_factory=lambda: [512, 256])
    actor_lr: float = 3e-5
    actor_entropy_scale: float = 3e-4
    actor_grad_clip: float = 100.0
    actor_min_std: float = 0.1
    actor_max_std: float = 1.0

    # ── Critic ──────────────────────────────────────────────────────────
    critic_hidden: list[int] = field(default_factory=lambda: [512, 256])
    critic_lr: float = 3e-5
    critic_grad_clip: float = 100.0
    slow_critic_update_freq: int = 100
    slow_critic_fraction: float = 0.02  # EMA rate for slow-target sync

    # ── Lambda-return ───────────────────────────────────────────────────
    gamma: float = 0.997
    lambda_: float = 0.95
    imagination_horizon: int = 15

    # ── Training loop ───────────────────────────────────────────────────
    prefill_steps: int = 2000
    train_ratio: int = 1        # WM + AC updates per collected env step
    batch_size: int = 16
    batch_seq_len: int = 64     # RSSM sequence length per batch
    buffer_capacity: int = 100_000

    # ── HER ─────────────────────────────────────────────────────────────
    her_enabled: bool = True
    her_ratio: float = 0.5
    her_success_reward: float = 10.0

    # ── Logging ─────────────────────────────────────────────────────────
    log_every_steps: int = 1000
    eval_every_steps: int = 10_000
    eval_episodes: int = 5

    # ── Misc ────────────────────────────────────────────────────────────
    seed: int = 0
    device: str = "cuda:0"
    logdir: str = "training/results/p3"
    wandb_project: str = "bantai-warehouse"
    wandb_mode: str = "online"  # "disabled" for no-internet / CI runs
