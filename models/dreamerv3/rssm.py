"""DreamerV3 RSSM: Recurrent State-Space Model with categorical stochastic.

Architecture follows NM512 vendor RSSM:
  - GRU deterministic path: deter=512
  - Categorical stochastic: 32 categories × 32 classes = 1024
  - Feature = cat(stoch_flat, deter) = 1024 + 512 = 1536
  - Straight-through gradients for categorical sampling
  - Unimix: 1% uniform for exploration
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd


class GRUCell(nn.Module):
    def __init__(self, inp_size: int, size: int, update_bias: float = -1.0):
        super().__init__()
        self._size = size
        self._update_bias = update_bias
        self.linear = nn.Linear(inp_size + size, 3 * size, bias=False)
        self.norm = nn.LayerNorm(3 * size, eps=1e-3)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        parts = self.norm(self.linear(torch.cat([x, h], -1)))
        reset, cand, update = parts.chunk(3, -1)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        return update * cand + (1 - update) * h


class RSSM(nn.Module):
    def __init__(self, stoch: int = 32, deter: int = 512, hidden: int = 512,
                 discrete: int = 32, num_actions: int = 6, embed_dim: int = 4352,
                 unimix: float = 0.01, device: str = "cuda:0"):
        super().__init__()
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._discrete = discrete
        self._unimix = unimix
        self._device = device

        stoch_dim = stoch * discrete

        # Prior: prev_stoch_flat + action → GRU input
        self._img_in = nn.Sequential(
            nn.Linear(stoch_dim + num_actions, hidden, bias=False),
            nn.LayerNorm(hidden, eps=1e-3), nn.SiLU(),
        )
        self._cell = GRUCell(hidden, deter)

        # Prior output: deter → logits
        self._img_out = nn.Sequential(
            nn.Linear(deter, hidden, bias=False),
            nn.LayerNorm(hidden, eps=1e-3), nn.SiLU(),
        )
        self._img_stat = nn.Linear(hidden, stoch * discrete)

        # Posterior: deter + embed → logits
        self._obs_out = nn.Sequential(
            nn.Linear(deter + embed_dim, hidden, bias=False),
            nn.LayerNorm(hidden, eps=1e-3), nn.SiLU(),
        )
        self._obs_stat = nn.Linear(hidden, stoch * discrete)

    @property
    def feat_dim(self) -> int:
        return self._stoch * self._discrete + self._deter

    def initial(self, batch_size: int) -> dict[str, torch.Tensor]:
        return {
            "deter": torch.zeros(batch_size, self._deter, device=self._device),
            "stoch": torch.zeros(batch_size, self._stoch, self._discrete, device=self._device),
            "logit": torch.zeros(batch_size, self._stoch, self._discrete, device=self._device),
        }

    def get_feat(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        stoch = state["stoch"].reshape(state["stoch"].shape[0], -1)
        return torch.cat([stoch, state["deter"]], -1)

    def feat_to_state(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        sd = self._stoch * self._discrete
        stoch_flat = feat[:, :sd]
        deter = feat[:, sd:]
        stoch = stoch_flat.reshape(-1, self._stoch, self._discrete)
        return {"stoch": stoch, "deter": deter, "logit": torch.zeros_like(stoch)}

    def _sample_stoch(self, logit: torch.Tensor) -> torch.Tensor:
        # Categorical with straight-through + unimix
        if self._unimix > 0:
            probs = F.softmax(logit, dim=-1)
            uniform = torch.ones_like(probs) / self._discrete
            probs = (1 - self._unimix) * probs + self._unimix * uniform
            logit = torch.log(probs + 1e-8)
        # Straight-through: sample one-hot, pass gradient through softmax
        dist = torchd.one_hot_categorical.OneHotCategoricalStraightThrough(logits=logit)
        return dist.rsample()

    def img_step(self, prev_state: dict[str, torch.Tensor],
                 prev_action: torch.Tensor) -> dict[str, torch.Tensor]:
        stoch_flat = prev_state["stoch"].reshape(prev_state["stoch"].shape[0], -1)
        x = torch.cat([stoch_flat, prev_action], -1)
        x = self._img_in(x)
        deter = self._cell(x, prev_state["deter"])
        x = self._img_out(deter)
        logit = self._img_stat(x).reshape(-1, self._stoch, self._discrete)
        stoch = self._sample_stoch(logit)
        return {"stoch": stoch, "deter": deter, "logit": logit}

    def obs_step(self, prev_state: dict[str, torch.Tensor],
                 prev_action: torch.Tensor,
                 embed: torch.Tensor) -> tuple[dict, dict]:
        prior = self.img_step(prev_state, prev_action)
        x = torch.cat([prior["deter"], embed], -1)
        x = self._obs_out(x)
        logit = self._obs_stat(x).reshape(-1, self._stoch, self._discrete)
        stoch = self._sample_stoch(logit)
        post = {"stoch": stoch, "deter": prior["deter"], "logit": logit}
        return post, prior

    def observe_sequence(self, embed: torch.Tensor, action: torch.Tensor,
                         is_first: torch.Tensor,
                         state: dict[str, torch.Tensor] | None = None
                         ) -> tuple[dict, dict]:
        """Process a (B, T, ...) sequence through the RSSM.

        Returns dicts of (B, T, ...) posterior and prior states.
        """
        B, T = embed.shape[:2]
        if state is None:
            state = self.initial(B)
        posts, priors = {k: [] for k in state}, {k: [] for k in state}

        for t in range(T):
            if t == 0 or is_first[:, t].any():
                mask = is_first[:, t].unsqueeze(-1).float()
                init = self.initial(B)
                for k in state:
                    shape = [1] * (state[k].ndim - 1)
                    m = mask.reshape(-1, *shape)
                    state[k] = state[k] * (1 - m) + init[k] * m

            act_t = action[:, t]
            post, prior = self.obs_step(state, act_t, embed[:, t])
            state = post
            for k in post:
                posts[k].append(post[k])
                priors[k].append(prior[k])

        posts = {k: torch.stack(v, dim=1) for k, v in posts.items()}
        priors = {k: torch.stack(v, dim=1) for k, v in priors.items()}
        return posts, priors

    def kl_loss(self, post: dict, prior: dict,
                free: float = 1.0, dyn_scale: float = 0.5,
                rep_scale: float = 0.1) -> torch.Tensor:
        post_logits = post["logit"]
        prior_logits = prior["logit"]
        post_probs = F.softmax(post_logits, dim=-1)
        prior_probs = F.softmax(prior_logits, dim=-1)
        # KL per category, sum over discrete dim
        kl = (post_probs * (torch.log(post_probs + 1e-8) - torch.log(prior_probs + 1e-8))).sum(-1)
        # Sum over stoch categories
        kl = kl.sum(-1)
        kl = torch.clamp(kl, min=free)
        # Dual KL: dyn_scale * KL(sg(post)||prior) + rep_scale * KL(post||sg(prior))
        post_probs_sg = post_probs.detach()
        prior_probs_sg = prior_probs.detach()
        dyn_kl = (post_probs_sg * (torch.log(post_probs_sg + 1e-8) - torch.log(prior_probs + 1e-8))).sum(-1).sum(-1)
        rep_kl = (post_probs * (torch.log(post_probs + 1e-8) - torch.log(prior_probs_sg + 1e-8))).sum(-1).sum(-1)
        dyn_kl = torch.clamp(dyn_kl, min=free)
        rep_kl = torch.clamp(rep_kl, min=free)
        loss = dyn_scale * dyn_kl + rep_scale * rep_kl
        return loss, kl
