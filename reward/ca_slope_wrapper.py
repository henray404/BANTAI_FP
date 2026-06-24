# reward/ca_slope_wrapper.py
# Person 5 — CA-SLOPE injected as a gymnasium env wrapper (the RECOMMENDED integration).
#
# WHY A WRAPPER: it adds the CA-SLOPE shaping term to the env reward WITHOUT touching P1 (env),
# P2 (world model) or P3 (policy/buffer/train loop). The friend running DreamerV3 just wraps the env
# once and passes it to P3Trainer unchanged — the shaped reward flows through env.step -> buffer.add
# -> world-model reward head -> imagination, exactly like any other reward.
#
#   from env.warehouse_env import WarehouseGymEnv, WarehouseEnvCfg
#   from reward.ca_slope_wrapper import CASlopeEnvWrapper
#   env = CASlopeEnvWrapper(WarehouseGymEnv(WarehouseEnvCfg()), mode="category")  # or "generic"/"none"
#   P3Trainer(env, world_model, cfg=P3Config()).run(200_000)
#
# It reads the UNANNEALED env.goal_pos (not the obs "goal" key, which anneals to zeros in curriculum
# stage 4) plus env.ee_pos / box_pos / holding / goal_id_buf — all owned by WarehouseRLEnv.

"""Gymnasium env wrapper that adds CA-SLOPE potential-based shaping to the per-step reward."""

from __future__ import annotations

from reward.ca_slope import CASlopeShaper


def _resolve_env(env):
    """Find the object carrying the warehouse state buffers (goal_pos etc).

    WarehouseGymEnv keeps them on its inner ._env (WarehouseRLEnv); also try the env itself and its
    gymnasium .unwrapped. Raises a clear error if none expose goal_pos.
    """
    for cand in (env, getattr(env, "unwrapped", None), getattr(env, "_env", None)):
        if cand is not None and hasattr(cand, "goal_pos"):
            return cand
    raise AttributeError(
        "CASlopeEnvWrapper could not find the state buffers (goal_pos, ee_pos, box_pos, holding, "
        "goal_id_buf). Wrap the WarehouseGymEnv (its ._env owns them)."
    )


class CASlopeEnvWrapper:
    """Add F = gamma*Phi(s') - Phi(s) to the env reward. Transparent to everything downstream.

    Args:
        env:  a WarehouseGymEnv (or anything exposing ee_pos/box_pos/goal_pos/holding/goal_id_buf).
        shaper: a configured CASlopeShaper; if None one is built from `mode`.
        mode: "category" (CA-SLOPE), "generic" (one gain), or "none" (no shaping — passthrough).
              Lets the RQ2 ablation toggle from a single flag without rebuilding the env.
    """

    def __init__(self, env, shaper: CASlopeShaper | None = None, mode: str = "category"):
        """Wrap `env`. mode picks the ablation arm when no explicit shaper is given."""
        if mode not in ("category", "generic", "none"):
            raise ValueError(f"mode must be category|generic|none, got {mode!r}")
        self.env = env
        self.mode = mode
        self._src = _resolve_env(env)
        if shaper is None:
            shaper = CASlopeShaper(category_aware=(mode == "category"))
        self.shaper = shaper
        self._prev_phi = None

    # Transparent passthrough for num_envs, device, action_space, render, close, etc.
    def __getattr__(self, name):
        """Delegate unknown attributes to the wrapped env (so it stays a drop-in)."""
        return getattr(self.env, name)

    def _potential(self):
        """Current CA-SLOPE potential Phi(s) (num_envs,) from the live env state buffers."""
        s = self._src
        # FRAME FIX (2026-06-24): box_pos is env-local WORLD; ee_pos is BASE-frame (a small delta),
        # so dist(ee_pos, box_pos) mixed frames -> wrong/garbage potential (same class as the C1 bug
        # fixed in reward_pickup.approach_box_distance). Use ee_pos_world to match box_pos's frame.
        ee = getattr(s, "ee_pos_world", s.ee_pos)
        return self.shaper.potential(ee, s.box_pos, s.goal_pos, s.holding, s.goal_id_buf)

    def reset(self, *args, **kwargs):
        """Reset the env and re-anchor Phi(s) so the first shaping step is well-defined."""
        out = self.env.reset(*args, **kwargs)
        self._prev_phi = None if self.mode == "none" else self._potential()
        return out

    def step(self, action):
        """Step the env and add the CA-SLOPE shaping term to the reward."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.mode == "none":
            return obs, reward, terminated, truncated, info

        phi = self._potential()
        if self._prev_phi is None:
            self._prev_phi = phi
        # F = gamma*Phi(s') - Phi(s), the smooth per-step PBRS term only.
        # NO terminal -Phi(s) bonus (was: blended on terminated|truncated). CRITICAL (2026-06-24):
        # Phi is negative-definite (= -gain*remaining), so the old -Phi(s) terminal term was a LARGE
        # POSITIVE one-shot (~+30 with phase_b_offset=13, gain=2) paid on EVERY episode end far from
        # goal -- timeout (truncated), crash/bounds/stuck (terminated). It dwarfed deliver(+15) and
        # failure_penalty(-15), so the agent learned to linger to the 1000-step cap / crash for the
        # bonus: return collapse, length pinned at the cap, actor stops exploring. On a GENUINE
        # success Phi~=0 (box at zone), so dropping the convention costs ~0 there. Timeouts must
        # BOOTSTRAP, not terminate; failures are already priced by failure_penalty in the base reward.
        f = self.shaper.gamma * phi - self._prev_phi
        self._prev_phi = phi

        shaped = reward + f
        if isinstance(info, dict):
            info = {**info, "ca_slope_shaping": f}
        return obs, shaped, terminated, truncated, info
