"""Pure-tensor staged pickup reward + delivery termination. No Isaac import (unit-testable).

Functions are duck-typed on a runtime env that exposes these buffers (populated by
WarehouseRLEnv / WarehouseGymEnv each step):
    env.ee_pos (N,3), env.box_pos (N,3), env.holding (N,) bool, env.goal_pos (N,3),
    env.grasp_event (N,) bool, env.drop_event (N,) bool.
They only use torch, so they work with any object carrying those attributes.
"""

from __future__ import annotations

import torch

DELIVER_RADIUS_M = 1.5  # box within this xy-distance of the goal zone center = delivered


def approach_box_distance(env) -> torch.Tensor:
    """Phase A dense: distance(ee, box), zero while holding (use with negative weight).

    Uses env.ee_pos_world (env-local world), NOT env.ee_pos (base-frame delta) — both ee and box
    must be in the SAME frame or the distance never shrinks on approach (dead gradient). See C1 in
    docs/project/training_readiness_2026-06-22.md.
    """
    d = torch.norm(env.ee_pos_world - env.box_pos, dim=-1)
    return torch.where(env.holding, torch.zeros_like(d), d)


def carry_distance(env) -> torch.Tensor:
    """Phase B dense: xy-distance(box, goal), zero while NOT holding (use with negative weight)."""
    d = torch.norm(env.box_pos[:, :2] - env.goal_pos[:, :2], dim=-1)
    return torch.where(env.holding, d, torch.zeros_like(d))


def grasp_success_reward(env) -> torch.Tensor:
    """+1 on the step grasp succeeds (one-shot). Use with positive weight (e.g. 5.0)."""
    return env.grasp_event.float()


def drop_penalty(env) -> torch.Tensor:
    """+1 on the step the box is dropped outside a zone (one-shot). Use with negative weight."""
    return env.drop_event.float()


def box_dropped(env, grace_steps: int = 5) -> torch.Tensor:
    """(N,) bool: a carried box fell to the floor this step (env.drop_event). Termination → reset.

    Suppressed during the spawn-settle grace window (matches warehouse_reward.RESET_GRACE_STEPS) so a
    step-0 weld/teleport transient (e.g. stage-1 pregrasp) can't false-trip it. Pure: reads only
    env.drop_event + env.episode_length_buf (no Isaac import).
    """
    n = getattr(env, "episode_length_buf", None)
    past = True if n is None else (n >= grace_steps)
    return env.drop_event & past


def carry_regress_penalty(env, regress_steps: int = 50) -> torch.Tensor:
    """-1 once a HELD box has gone `regress_steps` control steps WITHOUT nearing its goal zone
    (backing up / dawdling on the way to the finish zone). Use with a POSITIVE weight (like idle) →
    per-step cost until it makes progress again. Reads env._carry_regress_steps (set each step by
    WarehouseRLEnv._update_carry_progress); only counts while holding, so the approach phase (which
    heads to the box, not the zone) is never penalised."""
    steps = getattr(env, "_carry_regress_steps", None)
    if steps is None:
        return torch.zeros(env.num_envs, device=env.device)
    return -(steps >= regress_steps).float()


def pickup_delivered(env) -> torch.Tensor:
    """(N,) bool: holding AND box xy within DELIVER_RADIUS_M of the goal zone center."""
    in_zone = torch.norm(env.box_pos[:, :2] - env.goal_pos[:, :2], dim=-1) < DELIVER_RADIUS_M
    return env.holding & in_zone


def pickup_delivered_reward(env) -> torch.Tensor:
    """+1 per step while the held box is delivered in its zone (float of pickup_delivered)."""
    return pickup_delivered(env).float()


# ── Potential-based shaping (Ng/Russell 1999) ─────────────────────────────────
# Replaces the raw -w*dist dense terms (approach_box_distance / carry_distance), whose accumulation
# scaled with the 1000-step horizon (worst-case -78 / -182, see training_readiness_2026-06-22 C3).
# PBS reward F = γ·Φ(s') − Φ(s) with Φ = −dist telescopes: total over an episode ≈ Φ(start) − Φ(end)
# = d_start (bounded by the start distance), INDEPENDENT of step count. So a wandering robot no longer
# racks up unbounded dense cost, and the per-step gradient stays full strength (no freeze risk from
# shrinking the weight). γ MUST match the agent's discount or invariance breaks — keep them in sync.
PBS_GAMMA = 0.997  # == DreamerV3 discount (NM512 default 0.997). Change BOTH together.


def pbs_step(prev_dist: torch.Tensor, cur_dist: torch.Tensor, active_now: torch.Tensor,
             was_active: torch.Tensor, gamma: float = PBS_GAMMA):
    """One PBS step. F = γΦ(s')−Φ(s), Φ=−dist  →  F = prev_dist − γ·cur_dist (>0 when distance shrank).

    Emits shaping ONLY when the phase was active on BOTH this step and the previous one — that single
    guard suppresses the spike at episode reset AND at a phase boundary (grasp flips the active phase),
    where prev_dist is stale. Returns (shaping, new_prev_dist, new_active) for the caller to store.
    Pure tensor op (no Isaac) so the env's _update_pbs_shaping stays unit-testable.
    """
    f = prev_dist - gamma * cur_dist
    shaping = torch.where(active_now & was_active, f, torch.zeros_like(f))
    return shaping, cur_dist, active_now


def approach_box_shaped(env) -> torch.Tensor:
    """Phase-A PBS reward, Φ=−dist(ee,box). Reads env._approach_shaping (set once/step by
    WarehouseRLEnv._update_pbs_shaping — a pure READ so reward_debug re-calling it can't corrupt
    state). POSITIVE when nearing the box → use with a POSITIVE weight (sign flipped vs the old
    approach_box_distance, which used a negative weight on raw +distance)."""
    s = getattr(env, "_approach_shaping", None)
    return torch.zeros(env.num_envs, device=env.device) if s is None else s


def carry_shaped(env) -> torch.Tensor:
    """Phase-B PBS reward, Φ=−dist(box,zone). Reads env._carry_shaping. POSITIVE when nearing the
    zone → use with a POSITIVE weight (sign flipped vs the old carry_distance)."""
    s = getattr(env, "_carry_shaping", None)
    return torch.zeros(env.num_envs, device=env.device) if s is None else s


# ── Rack-avoidance shaping (mirrors the scripted demo's potential field) ───────
# scripts/demo_pickup.py KNOWS every rack xy and repels the base BEFORE contact (AVOID_INFLUENCE);
# the RL agent only ever sees collision/under_rack penalties AFTER it has already hit/entered a rack
# — too late to learn a smooth detour, so training often stalls grazing a sibling rack on the way to
# the box. These two terms hand the agent the SAME pre-contact gradient the demo steers by:
#   rack_avoid_shaped  : soft penalty ramping in as the chassis nears a NON-target rack (approach).
#   rack_backout_shaped: reward for INCREASING distance to the nearest rack while holding (escape the
#                        grab-rack right after grasp — mirrors the demo's BACKOUT_STEPS reverse).
# Both buffers are written once/step by WarehouseRLEnv._update_rack_avoid and only READ here, so the
# functions stay pure (same env-buffer-reader pattern as approach_box_shaped / carry_shaped).
RACK_AVOID_INFLUENCE_M = 1.6  # chassis-to-rack-centre distance under which the soft penalty ramps in


def rack_proximity_penalty(d_nearest: torch.Tensor, influence: float = RACK_AVOID_INFLUENCE_M):
    """Soft 0..1 ramp: 0 at/beyond `influence`, rising linearly to 1 at the rack centre. Returned
    NEGATIVE (use with a POSITIVE weight, like collision). Pure tensor op → unit-testable on its own."""
    pen = ((influence - d_nearest) / influence).clamp(0.0, 1.0)
    return -pen


def rack_avoid_shaped(env) -> torch.Tensor:
    """Approach-phase soft rack-proximity penalty. Reads env._rack_avoid (set by _update_rack_avoid;
    excludes the target rack so pressing in to grasp is never punished). NEGATIVE-when-near → use
    with a POSITIVE weight."""
    s = getattr(env, "_rack_avoid", None)
    return torch.zeros(env.num_envs, device=env.device) if s is None else s


def rack_backout_shaped(env) -> torch.Tensor:
    """Post-grasp escape reward: POSITIVE when a HELD robot moves AWAY from the nearest rack (backing
    out of the grab-rack). Reads env._rack_backout. POSITIVE weight."""
    s = getattr(env, "_rack_backout", None)
    return torch.zeros(env.num_envs, device=env.device) if s is None else s
