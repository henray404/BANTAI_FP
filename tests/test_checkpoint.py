# tests/test_checkpoint.py
# Person 5 — checkpoint milestone + auto-rewind rules, driven on the numpy toy env (Mac-runnable).

import numpy as np

from experiments.scenarios import DEFAULT_SCENARIOS
from experiments.toy_pickup_env import ToyPickupEnv
from recording.checkpoint import CheckpointManager


def _mgr(**kw):
    """CheckpointManager wired to the toy env capture/restore, with fast idle window for tests."""
    base = dict(
        capture_fn=lambda e: e.capture(),
        restore_fn=lambda e, b: e.restore(b),
        control_hz=10.0, idle_seconds=1.0,   # idle_steps = 10
        period_steps=10_000,                 # disable periodic during these tests
        approach_radius=1.0, move_eps=0.15,
    )
    base.update(kw)
    return CheckpointManager(**base)


def _env():
    """Toy env reset to the heavy scenario."""
    env = ToyPickupEnv(max_steps=10_000)
    env.reset(DEFAULT_SCENARIOS[2], seed=0)
    return env


def test_reset_takes_start_checkpoint():
    env, m = _env(), _mgr()
    m.reset(env)
    assert len(m.checkpoints) == 1
    assert m.nearest().label == "start"


def test_milestone_snapshots_and_nearest_is_most_recent():
    env, m = _env(), _mgr()
    m.reset(env)
    # near the box, not holding -> approach_target
    out = m.observe(env, step=1, base_xy=(6.0, 1.5), yaw=0.0, holding=False,
                    dist_box=0.5, dist_goal=13.0)
    assert out.snapshot_label == "approach_target"
    # grasp event -> grasp checkpoint, becomes the nearest
    out = m.observe(env, step=2, base_xy=(6.0, 1.0), yaw=0.0, holding=True,
                    dist_box=0.1, dist_goal=13.0, grasp_event=True)
    assert out.snapshot_label == "grasp"
    # carry near zone -> approach_delivery
    out = m.observe(env, step=3, base_xy=(6.0, -11.0), yaw=0.0, holding=True,
                    dist_box=0.0, dist_goal=0.8)
    assert out.snapshot_label == "approach_delivery"
    assert [c.label for c in m.checkpoints] == ["start", "approach_target", "grasp", "approach_delivery"]
    assert m.nearest().label == "approach_delivery"


def test_idle_triggers_rewind_to_nearest():
    env, m = _env(), _mgr()
    m.reset(env)
    start_xy = env.base_xy.copy()
    # move the robot somewhere, then sit still for the whole idle window
    env.base_xy = np.array([3.0, 3.0])
    restored = False
    for step in range(1, 12):
        out = m.observe(env, step=step, base_xy=(3.0, 3.0), yaw=0.5, holding=False,
                        dist_box=5.0, dist_goal=13.0)
        if out.restored:
            restored = True
            assert out.reason == "idle"
            break
    assert restored
    # rewound to the 'start' checkpoint -> env state restored
    assert np.allclose(env.base_xy, start_xy)


def test_spinning_in_place_triggers_rewind():
    env, m = _env(), _mgr()
    m.reset(env)
    reason = None
    for step in range(1, 12):
        # position fixed, yaw keeps turning ~1 rad/step -> circling the same spot
        out = m.observe(env, step=step, base_xy=(2.0, 2.0), yaw=float(step), holding=False,
                        dist_box=5.0, dist_goal=13.0)
        if out.restored:
            reason = out.reason
            break
    assert reason == "spinning"


def test_collision_triggers_immediate_rewind():
    env, m = _env(), _mgr()
    m.reset(env)
    out = m.observe(env, step=1, base_xy=(1.0, 1.0), yaw=0.0, holding=False,
                    dist_box=5.0, dist_goal=13.0, contact_force=12.0)
    assert out.restored and out.reason == "collision"
    assert out.checkpoint_step == 0  # nearest = start


def test_progress_checkpoint_only_when_getting_closer():
    env, m = _env(), _mgr(progress_delta=1.0)
    m.reset(env)
    # carrying, box approaching the zone. step1 sets the phase-B baseline; each later >=1m closer
    # step yields a progress checkpoint; a <1m step does not.
    labels = []
    for step, dg in enumerate([12.0, 11.0, 10.0, 9.5], start=1):
        out = m.observe(env, step=step, base_xy=(6.0, dg - 12.0), yaw=0.0, holding=True,
                        dist_box=0.0, dist_goal=dg)
        labels.append(out.snapshot_label)
    assert labels == [None, "progress", "progress", None]
    n_after_progress = len(m.checkpoints)  # start + 2 progress = 3

    # now move AWAY from the zone -> NO new checkpoints (nearest stays the best-progress state)
    for step, dg in enumerate([10.5, 12.0, 13.5], start=10):
        out = m.observe(env, step=step, base_xy=(6.0, dg - 12.0), yaw=0.0, holding=True,
                        dist_box=0.0, dist_goal=dg)
        assert out.snapshot_label is None
    assert len(m.checkpoints) == n_after_progress  # unchanged while receding


def test_no_periodic_checkpoint_by_default():
    env, m = _env(), _mgr(period_steps=0)  # progress-based only
    m.reset(env)
    for step in range(1, 200):
        # holding but distance to zone CONSTANT (no progress) -> no progress/periodic checkpoints
        out = m.observe(env, step=step, base_xy=(0.0, 0.0), yaw=0.0, holding=True,
                        dist_box=0.0, dist_goal=8.0)
        assert out.snapshot_label is None


def test_moving_normally_does_not_trigger():
    env, m = _env(), _mgr()
    m.reset(env)
    triggered = False
    for step in range(1, 30):
        # moving forward AND getting closer each step -> no idle, no no-progress
        out = m.observe(env, step=step, base_xy=(step * 0.5, 0.0), yaw=0.0, holding=False,
                        dist_box=20.0 - step * 0.5, dist_goal=13.0)
        triggered = triggered or out.restored
    assert not triggered


def test_drop_event_triggers_rewind():
    env, m = _env(), _mgr()
    m.reset(env)
    out = m.observe(env, step=1, base_xy=(1.0, 1.0), yaw=0.0, holding=False,
                    dist_box=2.0, dist_goal=10.0, drop_event=True)
    assert out.restored and out.reason == "drop"


def test_out_of_bounds_triggers_rewind():
    env, m = _env(), _mgr()
    m.reset(env)
    out = m.observe(env, step=1, base_xy=(50.0, 1.0), yaw=0.0, holding=False,
                    dist_box=2.0, dist_goal=10.0, out_of_bounds=True)
    assert out.restored and out.reason == "out_of_bounds"


def test_no_progress_triggers_when_moving_but_not_closer():
    # moving (so NOT idle) but distance never improves -> no_progress after the window
    env, m = _env(), _mgr(no_progress_seconds=1.0)  # no_progress_steps = 10
    m.reset(env)
    reason = None
    for step in range(1, 14):
        out = m.observe(env, step=step, base_xy=(step * 1.0, 0.0), yaw=0.0, holding=True,
                        dist_box=0.0, dist_goal=8.0)  # dist_goal constant -> no improvement
        if out.restored:
            reason = out.reason
            break
    assert reason == "no_progress"


def test_escalation_drops_unhelpful_checkpoint():
    env, m = _env(), _mgr(progress_delta=1.0, escalate_after=2, cooldown_steps=0)
    m.reset(env)
    # build a progress checkpoint at step 2 (carry, 10 -> 8 = 2m closer)
    m.observe(env, step=1, base_xy=(0.0, 0.0), yaw=0.0, holding=True, dist_box=0.0, dist_goal=10.0)
    m.observe(env, step=2, base_xy=(0.0, 1.0), yaw=0.0, holding=True, dist_box=0.0, dist_goal=8.0)
    assert m.nearest().label == "progress"
    # repeated collisions at the same spot: first two rewind to 'progress', the third escalates to 'start'
    steps_restored = []
    for step in range(3, 6):
        out = m.observe(env, step=step, base_xy=(0.0, 1.0), yaw=0.0, holding=True,
                        dist_box=0.0, dist_goal=8.0, contact_force=12.0)
        steps_restored.append(out.checkpoint_step)
    assert steps_restored == [2, 2, 0]  # progress, progress, then fell back to start
