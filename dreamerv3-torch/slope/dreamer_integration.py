# slope/dreamer_integration.py
# Integrasi SLOPERewardHead ke DreamerV3 WorldModel
# Approach: subclass WorldModel, override reward head + loss computation
# Flag use_slope=True/False untuk ablation

import torch
import torch.nn as nn
from slope.reward_head import SLOPERewardHead

to_np = lambda x: x.detach().cpu().numpy()


class SLOPEWorldModel(nn.Module):
    """
    Wrapper DreamerV3 WorldModel dengan SLOPE reward head.

    Cara pakai:
        base_wm   = WorldModel(obs_space, act_space, step, config)
        slope_wm  = SLOPEWorldModel(base_wm, config)

    Flag ablation di config:
        config.use_slope = True   -> pakai SLOPERewardHead + QCE loss
        config.use_slope = False  -> pakai reward head asli (MLP + log_prob)
    """

    def __init__(self, world_model, config):
        super().__init__()
        self.wm = world_model
        self._config = config
        self.use_slope = getattr(config, "use_slope", True)

        if self.use_slope:
            # Hitung feat_size sama seperti di WorldModel.__init__
            if config.dyn_discrete:
                feat_size = (
                    config.dyn_stoch * config.dyn_discrete + config.dyn_deter
                )
            else:
                feat_size = config.dyn_stoch + config.dyn_deter

            self.slope_head = SLOPERewardHead(
                input_dim=feat_size,
                hidden_dim=config.units,
                num_quantiles=getattr(config, "slope_num_quantiles", 32),
                gamma=getattr(config, "discount", 0.99),
            ).to(config.device)

            print(
                f"[SLOPE] SLOPERewardHead aktif — "
                f"feat_size={feat_size}, "
                f"num_quantiles={getattr(config, 'slope_num_quantiles', 32)}"
            )
        else:
            print("[SLOPE] use_slope=False — pakai reward head DreamerV3 asli")

    # ------------------------------------------------------------------
    # Forward: delegate ke base WorldModel
    # ------------------------------------------------------------------

    def __getattr__(self, name):
        """
        Delegate semua attribute yang tidak ada di SLOPEWorldModel
        ke base WorldModel (self.wm).
        Ini memastikan encoder, dynamics, heads, dll tetap accessible.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.wm, name)

    # ------------------------------------------------------------------
    # Train step dengan SLOPE reward loss
    # ------------------------------------------------------------------

    def train_step(self, data):
        """
        Override _train dengan SLOPE reward loss.

        Kalau use_slope=False, langsung delegate ke wm._train(data).
        Kalau use_slope=True, ganti reward loss dengan QCE loss dari SLOPE.

        Returns:
            post, context, metrics — sama seperti WorldModel._train
        """
        if not self.use_slope:
            return self.wm._train(data)

        # ---- Setup (sama seperti WorldModel._train) ----
        wm = self.wm
        config = self._config

        with wm._model_opt.context():  # handles grad + scaler
            with torch.cuda.amp.autocast(wm._use_amp):
                embed = wm.encoder(data)
                post, prior = wm.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                kl_free = config.kl_free
                dyn_scale = config.dyn_scale
                rep_scale = config.rep_scale
                kl_loss, kl_value, dyn_loss, rep_loss = wm.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )

                feat = wm.dynamics.get_feat(post)

                # ---- Losses dari semua head kecuali reward ----
                losses = {}
                preds = {}
                for name, head in wm.heads.items():
                    if name == "reward":
                        continue   # reward ditangani SLOPE di bawah
                    grad_head = name in config.grad_heads
                    f = feat if grad_head else feat.detach()
                    pred = head(f)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred

                for name, pred in preds.items():
                    loss = -pred.log_prob(data[name])
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss

                # ---- SLOPE reward loss (QCE) ----
                reward_feat = feat if "reward" in config.grad_heads else feat.detach()
                slope_result = self.slope_head.compute_loss(
                    reward_feat.reshape(-1, reward_feat.shape[-1]),  # flatten batch*time
                    data["reward"].reshape(-1),
                )
                # Reshape loss ke (batch, time) supaya assert shape konsisten
                slope_loss_shaped = slope_result["loss"].expand(embed.shape[:2])
                losses["reward"] = slope_loss_shaped

                # ---- Shaped reward untuk actor-critic (disimpan di context) ----
                with torch.no_grad():
                    # Ambil latent t dan t+1 untuk potential shaping
                    feat_flat = feat.reshape(-1, feat.shape[-1])
                    # Shift: t+1 approx dengan roll (sederhana, cukup untuk logging)
                    feat_next_flat = torch.roll(feat_flat, -1, dims=0)
                    reward_flat = data["reward"].reshape(-1)
                    shaped_r = self.slope_head.shaped_reward(
                        reward_flat, feat_flat, feat_next_flat
                    ).reshape(embed.shape[:2])

                # ---- Scaled losses + model loss ----
                scaled = {
                    key: value * wm._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss

            metrics = wm._model_opt(torch.mean(model_loss), wm.parameters())

        # ---- Metrics ----
        metrics.update(
            {f"{name}_loss": to_np(torch.mean(loss)) for name, loss in losses.items()}
        )
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(torch.mean(dyn_loss))
        metrics["rep_loss"] = to_np(torch.mean(rep_loss))
        metrics["kl"] = to_np(torch.mean(kl_value))

        # ---- W&B SLOPE metrics ----
        metrics["slope/potential"] = to_np(slope_result["potential"])
        metrics["slope/quantile_mean"] = to_np(slope_result["quantile_mean"])
        metrics["slope/reward_loss_qce"] = to_np(slope_result["loss"])
        metrics["slope/shaped_reward_mean"] = to_np(shaped_r.mean())

        with torch.cuda.amp.autocast(wm._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(wm.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(wm.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=feat,
                kl=kl_value,
                postent=wm.dynamics.get_dist(post).entropy(),
                shaped_reward=shaped_r,   # tersedia untuk actor-critic
            )

        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics
