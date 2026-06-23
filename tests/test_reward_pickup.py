import torch
from types import SimpleNamespace
from env.reward_pickup import (
    approach_box_distance, carry_distance, grasp_success_reward,
    drop_penalty, pickup_delivered, pickup_delivered_reward,
    pbs_step, approach_box_shaped, carry_shaped,
)


def _env(**kw):
    base = dict(
        num_envs=1, device="cpu",
        ee_pos_world=torch.tensor([[0.0, 0.0, 0.3]]),   # approach reward reads env-local world ee
        box_pos=torch.tensor([[0.0, 0.0, 0.3]]),
        holding=torch.tensor([False]),
        goal_pos=torch.tensor([[0.0, -12.0, 0.0]]),
        grasp_event=torch.tensor([False]),
        drop_event=torch.tensor([False]),
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_approach_distance_zero_when_ee_on_box():
    assert torch.allclose(approach_box_distance(_env()), torch.tensor([0.0]))


def test_approach_gated_off_when_holding():
    e = _env(holding=torch.tensor([True]), ee_pos_world=torch.tensor([[1.0, 0.0, 0.3]]))
    assert torch.allclose(approach_box_distance(e), torch.tensor([0.0]))  # gated -> 0


def test_carry_distance_active_only_when_holding():
    e = _env(holding=torch.tensor([True]),
             box_pos=torch.tensor([[0.0, 0.0, 0.3]]),
             goal_pos=torch.tensor([[0.0, -3.0, 0.0]]))
    assert torch.allclose(carry_distance(e), torch.tensor([3.0]))
    assert torch.allclose(carry_distance(_env()), torch.tensor([0.0]))  # not holding -> 0


def test_grasp_reward_fires_on_event():
    assert torch.allclose(grasp_success_reward(_env(grasp_event=torch.tensor([True]))),
                          torch.tensor([1.0]))


def test_drop_penalty_fires_on_event():
    assert torch.allclose(drop_penalty(_env(drop_event=torch.tensor([True]))),
                          torch.tensor([1.0]))


def test_pickup_delivered_requires_holding_and_in_zone():
    e = _env(holding=torch.tensor([True]),
             box_pos=torch.tensor([[0.0, -12.0, 0.3]]),
             goal_pos=torch.tensor([[0.0, -12.0, 0.0]]))
    assert bool(pickup_delivered(e)[0]) is True
    assert torch.allclose(pickup_delivered_reward(e), torch.tensor([1.0]))
    assert bool(pickup_delivered(_env())[0]) is False


# ── PBS ───────────────────────────────────────────────────────────────────────
def test_pbs_telescopes_to_start_distance():
    """With γ=1 the per-step shaping sums to Φ(start)−Φ(end) = d_start − d_end, INDEPENDENT of the
    path length — the whole point of PBS (bounded dense, no horizon accumulation)."""
    dists = [3.0, 2.5, 2.5, 1.0, 0.0]  # includes a no-progress step (2.5->2.5 contributes 0)
    prev = torch.tensor([dists[0]])
    active = torch.tensor([True])      # seed active so the first transition counts
    total = torch.zeros(1)
    for d in dists[1:]:
        f, prev, active = pbs_step(prev, torch.tensor([d]), torch.tensor([True]), active, gamma=1.0)
        total += f
    assert torch.allclose(total, torch.tensor([3.0]))  # d_start(3) − d_end(0), path-independent


def test_pbs_positive_on_progress_negative_on_regress():
    f, *_ = pbs_step(torch.tensor([3.0]), torch.tensor([2.0]), torch.tensor([True]),
                     torch.tensor([True]), gamma=1.0)
    assert float(f[0]) > 0                                            # got closer -> reward
    f, *_ = pbs_step(torch.tensor([2.0]), torch.tensor([3.0]), torch.tensor([True]),
                     torch.tensor([True]), gamma=1.0)
    assert float(f[0]) < 0                                            # backed up -> penalty


def test_pbs_suppresses_spike_at_entry():
    """was_active=False (reset / just-entered phase) -> 0 regardless of the stale prev_dist, so no
    step-1 reward spike."""
    f, _, active = pbs_step(torch.tensor([99.0]), torch.tensor([0.0]),
                            torch.tensor([True]), torch.tensor([False]), gamma=1.0)
    assert float(f[0]) == 0.0
    assert bool(active[0]) is True                                    # but now armed for next step


def test_pbs_inactive_phase_emits_zero():
    f, _, active = pbs_step(torch.tensor([3.0]), torch.tensor([2.0]),
                            torch.tensor([False]), torch.tensor([True]), gamma=1.0)
    assert float(f[0]) == 0.0                                         # phase not active -> no shaping
    assert bool(active[0]) is False


def test_shaped_funcs_read_env_buffers_and_fallback():
    e = _env(_approach_shaping=torch.tensor([0.12]), _carry_shaping=torch.tensor([0.34]))
    assert torch.allclose(approach_box_shaped(e), torch.tensor([0.12]))
    assert torch.allclose(carry_shaped(e), torch.tensor([0.34]))
    # missing buffer -> zeros (early init / non-warehouse env), never crashes
    assert torch.allclose(approach_box_shaped(_env()), torch.tensor([0.0]))
