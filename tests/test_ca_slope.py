# tests/test_ca_slope.py
# Person 5 — CA-SLOPE unit tests. Uses numpy (runs on a Mac with no torch); the module is
# backend-agnostic so the same asserts hold for torch tensors on the Isaac box.

import numpy as np

from reward.ca_slope import CASlopeShaper, DEFAULT_CATEGORY_GAINS, state_from_obs
from reward.ca_slope_wrapper import CASlopeEnvWrapper


def _state(ee, box, goal, holding, cat):
    """Build a batched (1, ...) CA-SLOPE state dict for one env."""
    oh = [0.0, 0.0, 0.0]
    oh[cat] = 1.0
    return {
        "ee_pos": np.array([ee], dtype=np.float64),
        "box_pos": np.array([box], dtype=np.float64),
        "goal_pos": np.array([goal], dtype=np.float64),
        "holding": np.array([float(holding)]),
        "goal_id": np.array([oh], dtype=np.float64),
    }


def test_gain_is_category_aware():
    s = CASlopeShaper(category_aware=True)
    gid = np.eye(3)  # one-hot rows for the 3 categories
    gains = s.gain(gid)
    assert np.allclose(gains, np.array(DEFAULT_CATEGORY_GAINS))


def test_generic_mode_collapses_gains():
    s = CASlopeShaper(category_aware=False, generic_gain=1.5)
    gid = np.eye(3)
    assert np.allclose(s.gain(gid), 1.5)  # same gain for every category


def test_potential_increases_as_ee_nears_box_phase_a():
    s = CASlopeShaper()
    far = s.potential(**_state([2, 0, 0.6], [0, 0, 0.6], [0, -12, 0], holding=False, cat=1))
    near = s.potential(**_state([0.1, 0, 0.6], [0, 0, 0.6], [0, -12, 0], holding=False, cat=1))
    assert float(near[0]) > float(far[0])  # closer to box -> higher (less negative) potential


def test_potential_continuous_at_grasp_flip():
    # phase_b_offset makes Phi near-continuous when holding flips with the box still far from goal.
    s = CASlopeShaper(phase_b_offset=13.0)
    ee_on_box = [6, 1, 0.72]
    box = [6, 1, 0.72]
    goal = [6, -12, 0]
    phi_pre = float(s.potential(**_state(ee_on_box, box, goal, holding=False, cat=2))[0])
    phi_post = float(s.potential(**_state(ee_on_box, box, goal, holding=True, cat=2))[0])
    # box->goal distance is 13, offset is 13 -> the two potentials match closely.
    assert abs(phi_pre - phi_post) < 1e-6


def test_shaping_potential_based_invariance_on_loop():
    # PBRS guarantee: shaping summed over a cycle returning to the start state telescopes to ~0
    # (gamma=1). Verify F over A->B->A nets to zero, i.e. shaping adds no net return to loops.
    s = CASlopeShaper(gamma=1.0)
    a = _state([2, 0, 0.6], [0, 0, 0.6], [0, -12, 0], holding=False, cat=0)
    b = _state([1, 0, 0.6], [0, 0, 0.6], [0, -12, 0], holding=False, cat=0)
    f_ab = float(s.shaping(a, b)[0])
    f_ba = float(s.shaping(b, a)[0])
    assert abs(f_ab + f_ba) < 1e-9


def test_terminal_shaping_uses_zero_next_potential():
    s = CASlopeShaper(gamma=0.99)
    a = _state([1, 0, 0.6], [0, 0, 0.6], [0, -12, 0], holding=False, cat=1)
    b = _state([0, 0, 0.6], [0, 0, 0.6], [0, -12, 0], holding=True, cat=1)
    phi_a = float(s.potential(**a)[0])
    f_term = float(s.shaping(a, b, done=np.array([1.0]))[0])
    assert np.isclose(f_term, -phi_a)  # Phi(terminal)=0 -> F = -Phi(s)


def test_shaping_batches():
    s = CASlopeShaper()
    prev = {
        "ee_pos": np.zeros((4, 3)), "box_pos": np.ones((4, 3)),
        "goal_pos": np.zeros((4, 3)), "holding": np.zeros(4),
        "goal_id": np.eye(3)[[0, 1, 2, 0]],
    }
    nxt = {k: (v + 0.1 if k != "goal_id" else v) for k, v in prev.items()}
    f = s.shaping(prev, nxt)
    assert np.asarray(f).shape == (4,)


# ── obs-dict helper (P3 buffer Batch path) ───────────────────────────────────

def test_state_from_obs_maps_goal_and_squeezes_holding():
    # v2 obs keys: 'goal' is the zone, holding is (B,1). state_from_obs maps + squeezes.
    obs = {
        "ee_pos": np.zeros((2, 3)), "box_pos": np.ones((2, 3)),
        "goal": np.full((2, 3), -12.0), "holding": np.array([[0.0], [1.0]]),
        "goal_id": np.eye(3)[[0, 1]],
    }
    st = state_from_obs(obs)
    assert st["goal_pos"].shape == (2, 3)          # 'goal' -> 'goal_pos'
    assert st["holding"].shape == (2,)             # (2,1) squeezed to (2,)
    # shaping_from_obs runs end-to-end with the (B,1) holding without a broadcast blow-up.
    f = CASlopeShaper().shaping_from_obs(obs, obs)
    assert np.asarray(f).shape == (2,)


# ── env wrapper (recommended P2/P3 integration) ──────────────────────────────

class _FakeWarehouseEnv:
    """Duck-typed stand-in exposing the WarehouseRLEnv buffers the wrapper reads (num_envs=1)."""

    def __init__(self):
        self.goal_pos = np.array([[0.0, -12.0, 0.0]])
        self.goal_id_buf = np.array([[0.0, 0.0, 1.0]])  # heavy -> gain 2.0
        self.box_pos = np.array([[0.0, 4.0, 0.6]])
        self.ee_pos = np.array([[0.0, 6.0, 0.6]])
        self.holding = np.array([False])

    def reset(self, **kw):
        return {"dummy": 0}, {}

    def step(self, action):
        self.ee_pos = self.ee_pos + np.array([[0.0, -1.0, 0.0]])  # approach the box
        return {"dummy": 0}, np.array([0.0]), np.array([False]), np.array([False]), {}


def test_wrapper_adds_potential_based_shaping_to_reward():
    env = _FakeWarehouseEnv()
    shaper = CASlopeShaper()
    w = CASlopeEnvWrapper(env, shaper=shaper, mode="category")
    w.reset()
    phi_prev = float(shaper.potential(env.ee_pos, env.box_pos, env.goal_pos,
                                      env.holding, env.goal_id_buf)[0])
    _, reward, _, _, info = w.step(np.zeros(6))
    phi_next = float(shaper.potential(env.ee_pos, env.box_pos, env.goal_pos,
                                      env.holding, env.goal_id_buf)[0])
    expected_f = shaper.gamma * phi_next - phi_prev
    assert np.isclose(float(np.asarray(reward)[0]), 0.0 + expected_f)  # base 0 + shaping
    assert np.isclose(float(np.asarray(info["ca_slope_shaping"])[0]), expected_f)


def test_wrapper_mode_none_is_passthrough():
    env = _FakeWarehouseEnv()
    w = CASlopeEnvWrapper(env, mode="none")
    w.reset()
    _, reward, _, _, info = w.step(np.zeros(6))
    assert float(np.asarray(reward)[0]) == 0.0          # unchanged base reward
    assert "ca_slope_shaping" not in info


def test_wrapper_terminal_uses_negative_prev_potential():
    class _TermEnv(_FakeWarehouseEnv):
        def step(self, action):
            self.ee_pos = self.ee_pos + np.array([[0.0, -1.0, 0.0]])
            return {"dummy": 0}, np.array([0.0]), np.array([True]), np.array([False]), {}

    env = _TermEnv()
    shaper = CASlopeShaper()
    w = CASlopeEnvWrapper(env, shaper=shaper, mode="category")
    w.reset()
    phi_prev = float(shaper.potential(env.ee_pos, env.box_pos, env.goal_pos,
                                      env.holding, env.goal_id_buf)[0])
    _, reward, term, _, _ = w.step(np.zeros(6))
    assert bool(np.asarray(term)[0])
    assert np.isclose(float(np.asarray(reward)[0]), -phi_prev)  # F = gamma*0 - Phi(s)
