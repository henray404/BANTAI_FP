# policy/train_loop.py
# P3 (Jeremy) — Training loop: env → buffer → WM (P2) → actor-critic (P3).
#
# Flow per step:
#   1. Collect one env step → buffer.add()
#   2. If buffer full enough:
#      a. buffer.sample() → wm.train_batch()  (P2 trains world model)
#      b. wm.encode_obs + wm.imagine_step() → train actor-critic via imagination
#   3. Episode done → buffer auto-applies Visual HER (EpisodeBuffer)
#   4. Log metrics to W&B via training.logger.Logger (P5)
#
# WorldModelInterface: ABC that P2 must implement.
# P3 calls wm.encode_obs() and wm.imagine_step() for imagination rollouts.
#
# Obs contract (v2, P1):  pixels, position, heading, goal, goal_id,
#                          ee_pos, gripper, holding, box_pos
# Action contract (P1):   (6,) [base_lin, base_ang, ee_dx, ee_dy, ee_dz, gripper]

"""P3 training loop with WorldModelInterface for DreamerV3 integration."""

from __future__ import annotations

import abc
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from buffer.replay_buffer import EpisodeBuffer, Batch
from buffer.visual_her import make_visual_her_fn, ZONE_POSITIONS
from policy.actor_critic import Actor, Critic, lambda_return
from policy.config import P3Config
from training.logger import Logger
from training.seed import seed_everything


# ── WorldModel Interface (P2 must implement) ──────────────────────────────────

class WorldModelInterface(abc.ABC):
    """Interface that P2's DreamerV3 world model must implement.

    P3 calls these methods in the imagination-based training loop.
    P2 fills the implementation using the NM512 vendored DreamerV3 code.

    RSSM feature = cat(deter, stoch.flatten()), shape (B, feat_dim).
    DreamerV3-small defaults: feat_dim = 512 + 32*32 = 1536.
    """

    @abc.abstractmethod
    def encode_obs(self, obs: dict, device: str = "cuda:0") -> Tensor:
        """Encode an obs dict into the current RSSM feature vector.

        Args:
            obs:    Dict with obs_v2 keys (numpy arrays, un-batched or (B,...) batch).
            device: Target device string.

        Returns:
            feat: (B, feat_dim) float32 RSSM feature.
        """

    @abc.abstractmethod
    def imagine_step(
        self,
        feat: Tensor,
        action: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Imagine one RSSM step (no observation).

        Args:
            feat:   (B, feat_dim) current RSSM feature.
            action: (B, action_dim) action to apply.

        Returns:
            next_feat:   (B, feat_dim) RSSM feature after imagined transition.
            pred_reward: (B,) predicted reward.
            pred_cont:   (B,) predicted continuation (1.0 = not done, 0.0 = done).
        """

    @abc.abstractmethod
    def train_batch(self, batch: Batch, device: str = "cuda:0") -> dict[str, float]:
        """Train the world model on a sampled batch.

        Args:
            batch:  Batch from EpisodeBuffer.sample().
            device: Target device string.

        Returns:
            metrics: Dict of scalar losses, e.g. {"wm/loss": 0.42, "wm/kl": 0.1}.
        """

    @abc.abstractmethod
    def get_feat_dim(self) -> int:
        """Return RSSM feature dimension. Must match P3Config.feat_dim."""

    def get_initial_feat(self, batch_size: int, device: str = "cuda:0") -> Tensor:
        """Return zeroed initial RSSM feature (default; P2 may override)."""
        return torch.zeros(batch_size, self.get_feat_dim(), device=device)


# ── Trainer ───────────────────────────────────────────────────────────────────

class P3Trainer:
    """Orchestrates the P3 DreamerV3 training loop.

    Responsibilities:
        - Collect env steps into EpisodeBuffer (Visual HER automatic).
        - Call P2's world model training on buffer samples.
        - Imagine H-step rollouts and train actor-critic via lambda-returns.
        - Log metrics to W&B (coordinates with P5 via training.logger).
        - Save / load checkpoints.

    Usage:
        trainer = P3Trainer(env, world_model, cfg=P3Config(seed=0))
        trainer.run(total_steps=200_000)

    Args:
        env:         WarehouseGymEnv (obs_v2 contract + action (6,)).
        world_model: WorldModelInterface from P2.
        cfg:         P3Config (optional, defaults to P3Config()).
    """

    def __init__(
        self,
        env,
        world_model: WorldModelInterface,
        cfg: Optional[P3Config] = None,
    ):
        self.cfg = cfg or P3Config()
        self.env = env
        self.wm = world_model
        self.device = self.cfg.device

        seed_everything(self.cfg.seed)

        # ── Buffer ────────────────────────────────────────────────────
        her_fn = (
            make_visual_her_fn(
                zone_positions=ZONE_POSITIONS,
                success_reward=self.cfg.her_success_reward,
                her_ratio=self.cfg.her_ratio,
            )
            if self.cfg.her_enabled
            else None
        )
        self.buffer = EpisodeBuffer(
            capacity=self.cfg.buffer_capacity,
            her_fn=her_fn,
            seed=self.cfg.seed,
        )

        # ── Actor / Critic ────────────────────────────────────────────
        feat_dim = world_model.get_feat_dim()
        if feat_dim != self.cfg.feat_dim:
            raise ValueError(
                f"WorldModel.get_feat_dim()={feat_dim} != P3Config.feat_dim="
                f"{self.cfg.feat_dim}. Align P2's RSSM config with P3Config."
            )

        self.actor = Actor(
            feat_dim=feat_dim,
            action_dim=self.cfg.action_dim,
            hidden=self.cfg.actor_hidden,
            min_std=self.cfg.actor_min_std,
            max_std=self.cfg.actor_max_std,
            entropy_scale=self.cfg.actor_entropy_scale,
        ).to(self.device)

        self.critic = Critic(
            feat_dim=feat_dim,
            hidden=self.cfg.critic_hidden,
            slow_target_freq=self.cfg.slow_critic_update_freq,
            slow_frac=self.cfg.slow_critic_fraction,
        ).to(self.device)

        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=self.cfg.actor_lr, eps=1e-5
        )
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=self.cfg.critic_lr, eps=1e-5
        )

        # ── Logger ────────────────────────────────────────────────────
        Path(self.cfg.logdir).mkdir(parents=True, exist_ok=True)
        self.logger = Logger(
            project=self.cfg.wandb_project,
            config=vars(self.cfg),
            name=f"p3_dreamer_seed{self.cfg.seed}",
            mode=self.cfg.wandb_mode,
            print_every=0,
        )

        # ── Runtime state ─────────────────────────────────────────────
        self._global_step = 0
        self._episode_reward = 0.0
        self._episode_len = 0
        self._episode_count = 0
        self._obs, _ = env.reset(seed=self.cfg.seed)
        self._feat = world_model.get_initial_feat(1, self.device)

        # ── Curriculum (P4 mechanism: WarehouseRLEnv.set_stage / set_goal_alpha) ──
        # P4 exposes the env API; P3 owns the transition policy (sliding success-rate gate below).
        self._succ_window: list[bool] = []
        self._stage = int(getattr(self.cfg, "curriculum_start_stage", 1))
        self._maybe_set_stage(self._stage)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, total_steps: int) -> None:
        """Run the training loop for `total_steps` env interactions.

        Args:
            total_steps: Total env steps before returning.
        """
        t0 = time.time()

        while self._global_step < total_steps:
            action, _ = self._select_action()
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = bool(terminated) or bool(truncated)

            self.buffer.add(self._obs, action, float(reward), next_obs, done)
            self._episode_reward += float(reward)
            self._episode_len += 1
            self._global_step += 1

            # Update running RSSM feature (observation-conditioned step).
            action_t = torch.as_tensor(
                action, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            self._feat, _, _ = self.wm.imagine_step(self._feat, action_t)

            if done:
                success = self._episode_reward > getattr(
                    self.cfg, "curriculum_success_reward", 5.0
                )
                self._advance_curriculum(success)
                self._log_episode(t0)
                self._obs, _ = self.env.reset()
                self._feat = self.wm.get_initial_feat(1, self.device)
                self._episode_reward = 0.0
                self._episode_len = 0
                self._episode_count += 1
            else:
                self._obs = next_obs

            # Training updates (only after prefill).
            wm_metrics = {}
            ac_metrics = {}
            if len(self.buffer) >= self.cfg.prefill_steps:
                for _ in range(self.cfg.train_ratio):
                    wm_metrics = self._train_world_model()
                    ac_metrics = self._train_actor_critic()

            if self._global_step % self.cfg.log_every_steps == 0:
                fps = self._global_step / max(time.time() - t0, 1.0)
                all_metrics = {**wm_metrics, **ac_metrics,
                               "train/fps": fps,
                               "train/buffer_size": len(self.buffer)}
                self.logger.log(all_metrics, step=self._global_step)

        self.logger.finish()

    def eval_episode(self) -> dict[str, float]:
        """Run one deterministic evaluation episode (no exploration noise).

        Returns:
            metrics: {"eval/reward": ..., "eval/length": ..., "eval/success": ...}
        """
        obs, _ = self.env.reset()
        feat = self.wm.get_initial_feat(1, self.device)
        ep_reward = 0.0
        ep_len = 0
        success = False

        while True:
            feat_t = feat
            with torch.no_grad():
                action = self.actor.mean_action(feat_t).squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, info = self.env.step(action)
            ep_reward += float(reward)
            ep_len += 1
            action_t = torch.as_tensor(
                action, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            feat, _, _ = self.wm.imagine_step(feat, action_t)
            if terminated:
                success = True
            if terminated or truncated:
                break

        return {"eval/reward": ep_reward, "eval/length": ep_len, "eval/success": float(success)}

    def save(self, path: Optional[str] = None) -> Path:
        """Save actor and critic weights. Returns the checkpoint path."""
        out = Path(path or self.cfg.logdir) / f"p3_step{self._global_step}.pt"
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "global_step": self._global_step,
        }, out)
        return out

    def load(self, path: str) -> None:
        """Restore actor and critic from a checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        self._global_step = ckpt.get("global_step", 0)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _select_action(self) -> tuple[np.ndarray, float]:
        """Return a numpy action and its log-prob scalar."""
        with torch.no_grad():
            if len(self.buffer) < self.cfg.prefill_steps:
                action_t = torch.rand(1, self.cfg.action_dim, device=self.device) * 2 - 1
                return action_t.squeeze(0).cpu().numpy(), 0.0
            action_t, lp = self.actor(self._feat)
        return action_t.squeeze(0).cpu().numpy(), float(lp.item())

    def _train_world_model(self) -> dict[str, float]:
        """Sample a batch and call P2's world-model train step."""
        batch = self.buffer.sample(self.cfg.batch_size)
        return self.wm.train_batch(batch, device=self.device)

    def _train_actor_critic(self) -> dict[str, float]:
        """Imagine H-step rollout from buffer-sampled start; update actor + critic."""
        H = self.cfg.imagination_horizon
        B = self.cfg.batch_size

        # Encode starting RSSM features from a buffer sample.
        batch = self.buffer.sample(B)
        start_feat = self.wm.encode_obs(batch.obs, device=self.device)  # (B, feat_dim)

        # ── Imagination rollout (detached from WM gradients) ──────────
        feats: list[Tensor] = [start_feat.detach()]
        rewards: list[Tensor] = []
        conts: list[Tensor] = []

        feat = start_feat.detach()
        for _ in range(H):
            with torch.no_grad():
                action_t, _ = self.actor(feat)
            next_feat, pred_r, pred_c = self.wm.imagine_step(feat, action_t)
            feats.append(next_feat.detach())
            rewards.append(pred_r.detach())
            conts.append(pred_c.detach())
            feat = next_feat.detach()

        feats_t = torch.stack(feats)      # (H+1, B, feat_dim)
        rewards_t = torch.stack(rewards)  # (H, B)
        conts_t = torch.stack(conts)      # (H, B)

        # Slow-target values for lambda-return targets.
        with torch.no_grad():
            vals_slow = self.critic.slow_value(
                feats_t.flatten(0, 1)
            ).squeeze(-1).view(H + 1, B)

        returns = lambda_return(
            rewards_t, vals_slow, conts_t,
            gamma=self.cfg.gamma, lambda_=self.cfg.lambda_,
        )  # (H, B)

        # ── Critic update ─────────────────────────────────────────────
        critic_loss = self.critic.loss(feats_t[:-1].detach(), returns)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.critic_grad_clip)
        self.critic_opt.step()
        self.critic.update_slow_target()

        # ── Actor update (re-imagine with gradients through actor) ────
        feat = start_feat.detach()
        feats_actor: list[Tensor] = []
        for _ in range(H):
            action_t, _ = self.actor(feat)            # gradients flow here
            next_feat, _, _ = self.wm.imagine_step(feat, action_t)
            feats_actor.append(feat)
            feat = next_feat.detach()

        feats_actor_t = torch.stack(feats_actor)      # (H, B, feat_dim)

        # Critic baseline V(feat) for the actor advantage (C1 variance reduction).
        with torch.no_grad():
            actor_values = self.critic(
                feats_actor_t.flatten(0, 1).detach()
            ).squeeze(-1).view(H, B)
        actor_loss = self.actor.loss(feats_actor_t, returns, actor_values)
        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.actor_grad_clip)
        self.actor_opt.step()

        return {
            "ac/actor_loss": float(actor_loss.item()),
            "ac/critic_loss": float(critic_loss.item()),
            "ac/mean_return": float(returns.mean().item()),
            "ac/mean_imag_reward": float(rewards_t.mean().item()),
        }

    def _log_episode(self, t0: float) -> None:
        self.logger.log({
            "ep/reward": self._episode_reward,
            "ep/length": self._episode_len,
            "ep/count": self._episode_count,
            "curriculum/stage": float(getattr(self, "_stage", 3)),
        }, step=self._global_step)

    # ── Curriculum (P4 mechanism, P3 transition policy) ─────────────────────────
    def _env_inner(self):
        """Underlying WarehouseRLEnv exposing the curriculum API (set_stage/set_goal_alpha)."""
        return getattr(self.env, "_env", self.env)

    def _maybe_set_stage(self, stage: int) -> None:
        """Set the env curriculum stage if enabled and supported (no-op otherwise)."""
        env = self._env_inner()
        if getattr(self.cfg, "curriculum_enabled", True) and hasattr(env, "set_stage"):
            env.set_stage(stage)
            self._stage = stage

    def _advance_curriculum(self, episode_success: bool) -> None:
        """Bump stage once a window of recent episodes succeeds; anneal goal xyz in stage 4.

        Sliding success-rate gate. All thresholds read from cfg with safe defaults so this is a
        no-op on envs without set_stage or when cfg.curriculum_enabled is False.
        """
        env = self._env_inner()
        if not (getattr(self.cfg, "curriculum_enabled", True) and hasattr(env, "set_stage")):
            return
        window = int(getattr(self.cfg, "curriculum_window", 20))
        thresh = float(getattr(self.cfg, "curriculum_success_threshold", 0.6))
        self._succ_window.append(bool(episode_success))
        if len(self._succ_window) > window:
            self._succ_window.pop(0)
        if (len(self._succ_window) >= window
                and sum(self._succ_window) / len(self._succ_window) >= thresh
                and self._stage < 3):
            self._maybe_set_stage(self._stage + 1)
            self._succ_window.clear()
        if self._stage >= 4 and hasattr(env, "set_goal_alpha"):
            anneal = float(getattr(self.cfg, "goal_anneal_steps", 50_000))
            env.set_goal_alpha(max(0.0, 1.0 - self._global_step / anneal))
