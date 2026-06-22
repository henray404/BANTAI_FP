"""Per-step reward breakdown — see WHICH reward term drives each step (debugging/understanding).

Re-runs each RewardsCfg term on the live RL env and returns its WEIGHTED contribution, so you can
answer "why is return negative?", "does grasp ever fire?", "is the approach pull actually shrinking?"
without reading the trainer internals. Used by scripts/drive_env.py --debug_reward; importable for a
quick eval probe too. No Isaac import here — duck-typed on the env's reward buffers.
"""

from __future__ import annotations

# Display order = the RewardsCfg term order (warehouse_env.RewardsCfg). Missing terms are skipped,
# so renaming a term degrades gracefully instead of crashing the teleop loop.
TERM_ORDER = ("approach", "grasp", "carry", "deliver", "time_pen", "collision", "idle", "idle_slow", "drop", "failure", "under_rack", "carry_regress")


def reward_breakdown(rl_env, env_idx: int = 0) -> dict[str, float]:
    """Return {term_name: weighted_reward} for one env, computed from rl_env.cfg.rewards.

    A term that raises (needs scene state not ready) is reported as NaN rather than crashing.
    """
    rewards_cfg = rl_env.cfg.rewards
    out: dict[str, float] = {}
    for name in TERM_ORDER:
        term = getattr(rewards_cfg, name, None)
        if term is None or not hasattr(term, "func") or not hasattr(term, "weight"):
            continue
        try:
            val = term.func(rl_env, **getattr(term, "params", {}) or {})
            out[name] = float(val[env_idx]) * float(term.weight)
        except Exception:
            out[name] = float("nan")
    return out


def format_breakdown(bd: dict[str, float]) -> str:
    """One compact line: 'approach=-0.31 grasp=0.00 ... TOTAL=-0.32' (NaN terms shown as 'na')."""
    total = sum(v for v in bd.values() if v == v)  # skip NaN (NaN != NaN)
    parts = " ".join(f"{k}={'na' if v != v else f'{v:+.3f}'}" for k, v in bd.items())
    return f"[reward] {parts} TOTAL={total:+.3f}"


if __name__ == "__main__":  # tiny self-check (no Isaac): buffer-only terms compute, scene term -> NaN
    from types import SimpleNamespace
    import torch
    from env.reward_pickup import approach_box_distance, grasp_success_reward

    def _bad(_env):  # stands in for a term needing scene state that isn't there
        raise RuntimeError("needs scene")

    env = SimpleNamespace(
        ee_pos_world=torch.tensor([[3.0, 0.0, 0.3]]),
        box_pos=torch.tensor([[0.0, 0.0, 0.3]]),
        holding=torch.tensor([False]),
        grasp_event=torch.tensor([True]),
        cfg=SimpleNamespace(rewards=SimpleNamespace(
            approach=SimpleNamespace(func=approach_box_distance, weight=-0.05, params={}),
            grasp=SimpleNamespace(func=grasp_success_reward, weight=5.0, params={}),
            collision=SimpleNamespace(func=_bad, weight=2.0, params={}),
        )),
    )
    bd = reward_breakdown(env)
    assert abs(bd["approach"] - (-0.05 * 3.0)) < 1e-6, bd          # 3 m away * -0.05
    assert abs(bd["grasp"] - 5.0) < 1e-6, bd                       # grasp fired * 5.0
    assert bd["collision"] != bd["collision"], bd                 # NaN (graceful)
    print(format_breakdown(bd))
    print("reward_debug self-check OK")
