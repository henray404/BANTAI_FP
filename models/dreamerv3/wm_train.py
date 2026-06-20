"""WarehouseWorldModel: DreamerV3 world model implementing WorldModelInterface.

Assembles encoder + RSSM + decoder + reward/cont heads.
Trains on Batch from EpisodeBuffer (single transitions).
Provides encode_obs / imagine_step for P3 actor-critic imagination.
"""

from __future__ import annotations

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np

from policy.train_loop import WorldModelInterface
from buffer.replay_buffer import Batch
from models.dreamerv3.encoder import Encoder, LOW_DIM_KEYS
from models.dreamerv3.rssm import RSSM
from models.dreamerv3.decoder import ConvDecoder, RewardHead, ContHead, symlog


class WarehouseWorldModel(WorldModelInterface, nn.Module):
    def __init__(self, obs_space, action_dim: int = 6, device: str = "cuda:0",
                 dyn_stoch: int = 32, dyn_deter: int = 512, dyn_discrete: int = 32,
                 cnn_depth: int = 32, lr: float = 3e-4):
        nn.Module.__init__(self)
        self._device = device
        self._action_dim = action_dim
        self._dyn_stoch = dyn_stoch
        self._dyn_deter = dyn_deter
        self._dyn_discrete = dyn_discrete

        self.encoder = Encoder(depth=cnn_depth)
        embed_dim = self.encoder.out_dim

        self.rssm = RSSM(
            stoch=dyn_stoch, deter=dyn_deter, hidden=dyn_deter,
            discrete=dyn_discrete, num_actions=action_dim,
            embed_dim=embed_dim, device=device,
        )

        feat_size = self.rssm.feat_dim
        self.decoder = ConvDecoder(feat_size, shape=(3, 64, 64), depth=cnn_depth)
        self.reward_head = RewardHead(feat_size)
        self.cont_head = ContHead(feat_size)

        self.to(device)
        self._opt = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-5)
        self._grad_clip = 100.0

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[WarehouseWorldModel] {n_params:,} parameters, feat_dim={feat_size}")

    def get_feat_dim(self) -> int:
        return self.rssm.feat_dim

    def get_initial_feat(self, batch_size: int, device: str = "cuda:0") -> Tensor:
        return torch.zeros(batch_size, self.rssm.feat_dim, device=device)

    def _obs_to_tensors(self, obs: dict, device: str) -> dict[str, Tensor]:
        out = {}
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                t = torch.from_numpy(v).float()
            elif isinstance(v, Tensor):
                t = v.float()
            else:
                t = torch.tensor(v, dtype=torch.float32)
            if t.ndim == len(self._obs_shape(k)):
                t = t.unsqueeze(0)
            out[k] = t.to(device)
        return out

    def _obs_shape(self, key: str) -> tuple:
        if key == "pixels":
            return (3, 64, 64)
        shapes = {"position": (3,), "heading": (2,), "goal": (3,), "goal_id": (3,),
                  "ee_pos": (3,), "gripper": (1,), "holding": (1,), "box_pos": (3,)}
        return shapes.get(key, ())

    def encode_obs(self, obs: dict, device: str = "cuda:0") -> Tensor:
        obs_t = self._obs_to_tensors(obs, device)
        with torch.no_grad():
            embed = self.encoder(obs_t)
            B = embed.shape[0]
            state = self.rssm.initial(B)
            zero_action = torch.zeros(B, self._action_dim, device=device)
            post, _ = self.rssm.obs_step(state, zero_action, embed)
            return self.rssm.get_feat(post)

    def imagine_step(self, feat: Tensor, action: Tensor
                     ) -> tuple[Tensor, Tensor, Tensor]:
        with torch.no_grad():
            state = self.rssm.feat_to_state(feat)
            next_state = self.rssm.img_step(state, action)
            next_feat = self.rssm.get_feat(next_state)
            pred_reward = self.reward_head(next_feat)
            pred_cont = torch.sigmoid(self.cont_head(next_feat))
            return next_feat, pred_reward, pred_cont

    def train_batch(self, batch: Batch, device: str = "cuda:0") -> dict[str, float]:
        obs_t = self._obs_to_tensors(batch.obs, device)
        next_obs_t = self._obs_to_tensors(batch.next_obs, device)
        action = torch.from_numpy(batch.action).float().to(device)
        reward = torch.from_numpy(batch.reward).float().to(device)
        done = torch.from_numpy(batch.done).float().to(device)
        cont_target = 1.0 - done

        # Encode current and next obs
        embed = self.encoder(obs_t)
        next_embed = self.encoder(next_obs_t)
        B = embed.shape[0]

        # RSSM: initial → obs_step(obs) → posterior_t
        init_state = self.rssm.initial(B)
        zero_action = torch.zeros(B, self._action_dim, device=device)
        post_t, _ = self.rssm.obs_step(init_state, zero_action, embed)

        # RSSM: posterior_t + action → prior_t+1 (imagination)
        prior_tp1 = self.rssm.img_step(post_t, action)

        # RSSM: posterior_t + action → obs_step(next_obs) → posterior_t+1
        post_tp1, _ = self.rssm.obs_step(post_t, action, next_embed)

        # KL loss between prior prediction and posterior (single-step)
        kl_loss, kl_value = self.rssm.kl_loss(post_tp1, prior_tp1)
        kl_loss = kl_loss.mean()

        # Decode from posterior features
        feat_t = self.rssm.get_feat(post_t)
        feat_tp1 = self.rssm.get_feat(post_tp1)

        # Image reconstruction loss (current obs)
        recon = self.decoder(feat_t)
        recon_loss = F.mse_loss(recon, obs_t["pixels"])

        # Reward prediction from next-state features
        rew_loss = self.reward_head.loss(feat_tp1, reward).mean()

        # Continuation prediction
        cont_loss = self.cont_head.loss(feat_tp1, cont_target).mean()

        # Total loss
        total_loss = kl_loss + recon_loss + rew_loss + cont_loss

        self._opt.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self._grad_clip)
        self._opt.step()

        return {
            "wm/loss": total_loss.item(),
            "wm/kl": kl_value.mean().item(),
            "wm/recon": recon_loss.item(),
            "wm/reward": rew_loss.item(),
            "wm/cont": cont_loss.item(),
        }
