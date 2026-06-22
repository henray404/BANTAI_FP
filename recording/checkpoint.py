# recording/checkpoint.py
# Person 5 — task-milestone state checkpoints + auto-rewind on stall / collision.
#
# DIFFERENT from a "policy checkpoint" (saved network weights). Here a checkpoint is a SNAPSHOT of
# the full sim state (all joints + box pose + flags) taken at task milestones, so that when the robot
# misbehaves we restore ("rewind") the nearest snapshot instead of failing the whole run.
#
# Snapshots are taken at:
#   - approach_target   : first time the EE/base gets within approach_radius of the box (not holding)
#   - grasp             : the step the box is grasped (holding flips 0 -> 1)
#   - approach_delivery : first time the carried box gets within approach_radius of the goal zone
#   - progress          : whenever the robot gets `progress_delta` metres CLOSER to its active target
#                         (box while approaching, final zone while carrying). NOT a fixed step
#                         counter: if the robot wanders AWAY, no checkpoint is saved, so the nearest
#                         checkpoint is always the best-progress state — never a worse, further one.
#   - periodic          : optional fixed-interval fallback, OFF by default (period_steps=0).
#
# A rewind to the NEAREST (most recent) snapshot is triggered by:
#   - collision    : chassis contact force over threshold (hit a rack / obstacle / box)
#   - drop         : the box is dropped mid-carry (rewind to where it was still held)
#   - out_of_bounds: the robot left the warehouse interior
#   - spinning     : base barely moved but yaw kept turning (circling the same spot)
#   - idle         : base barely moved for `idle_seconds` (robot stuck, doesn't know what to do)
#   - no_progress  : moving but not getting any closer to the target for `no_progress_seconds`
# If rewinding to the same checkpoint keeps failing, it is dropped and we fall back to an earlier one.
#
# BACKEND-AGNOSTIC: capture/restore of the actual state is delegated to capture_fn/restore_fn, so the
# same manager drives the numpy toy env (testable on a Mac) and the real Isaac env (sim writes).

"""Milestone state checkpoints with auto-rewind on idle / spinning / collision."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class Checkpoint:
    """One saved state snapshot at a task milestone."""

    step: int
    label: str
    blob: Any                 # opaque state from capture_fn (toy dict / sim tensors)
    base_xy: tuple[float, float]
    phase: str                # "A" (approach/grasp) or "B" (carry/place)


@dataclass
class StepOutcome:
    """What the manager did this step (for CSV logging)."""

    snapshot_label: Optional[str] = None   # set if a checkpoint was taken this step
    restored: bool = False
    reason: Optional[str] = None           # idle | spinning | collision (when restored)
    checkpoint_step: Optional[int] = None  # which checkpoint we rewound to


@dataclass
class CheckpointManager:
    """Take milestone snapshots and rewind to the nearest one on stall/collision.

    capture_fn(env) -> blob  and  restore_fn(env, blob) are the only env-touch points; everything
    else is pure bookkeeping so the rules are unit-testable without a simulator.
    """

    capture_fn: Callable[[Any], Any]
    restore_fn: Callable[[Any, Any], None]
    control_hz: float = 10.0
    idle_seconds: float = 30.0          # stuck this long -> rewind
    progress_delta: float = 2.5         # m closer to the active target before a new progress snapshot
                                        # (~2.5m suits a 20x30m warehouse; ~13m carry -> ~5 snapshots)
    period_steps: int = 0               # optional fixed-interval fallback (0 = off; progress-based only)
    approach_radius: float = 1.0        # m: "near the box" / "near the zone"
    move_eps: float = 0.15              # m: displacement below this over the window = "not moving"
    spin_yaw_threshold: float = 2.0 * math.pi  # total |yaw change| over window to call it spinning
    collision_force_n: float = 5.0      # matches env collision_penalty threshold
    no_progress_seconds: float = 45.0   # moving but not getting any closer this long -> rewind
    escalate_after: int = 3             # rewinds to the SAME checkpoint before dropping it (go earlier)
    cooldown_steps: int = 5             # don't re-trigger immediately after a rewind
    max_rewinds: int = 50               # safety budget across the whole run

    checkpoints: list[Checkpoint] = field(default_factory=list)
    _seen_approach: bool = False
    _seen_grasp: bool = False
    _seen_delivery: bool = False
    _last_prog_dist: Optional[float] = None   # active-target distance at the last snapshot
    _prog_phase: str = "A"                     # phase the progress baseline was measured in
    _best_active: Optional[float] = None       # best (smallest) active-target distance seen
    _no_improve: int = 0                        # steps since the active distance last improved
    _restore_counts: dict = field(default_factory=dict)  # per-checkpoint-step rewind tally (escalation)
    _xy_hist: deque = field(default_factory=deque)
    _yaw_hist: deque = field(default_factory=deque)
    _cooldown: int = 0
    _rewinds: int = 0

    @property
    def idle_steps(self) -> int:
        """Window length (steps) for the idle/spin detector."""
        return max(1, int(round(self.idle_seconds * self.control_hz)))

    @property
    def no_progress_steps(self) -> int:
        """Steps of no distance improvement before the no-progress rewind fires."""
        return max(1, int(round(self.no_progress_seconds * self.control_hz)))

    def reset(self, env) -> None:
        """Clear history and take a 'start' checkpoint at the spawn state."""
        self.checkpoints.clear()
        self._seen_approach = self._seen_grasp = self._seen_delivery = False
        self._last_prog_dist = None
        self._prog_phase = "A"
        self._best_active = None
        self._no_improve = 0
        self._restore_counts = {}
        self._xy_hist = deque(maxlen=self.idle_steps)
        self._yaw_hist = deque(maxlen=self.idle_steps)
        self._cooldown = 0
        self._rewinds = 0
        self._snapshot(env, step=0, label="start", base_xy=(0.0, 0.0), phase="A")

    def _snapshot(self, env, *, step, label, base_xy, phase) -> None:
        """Capture and store one checkpoint."""
        self.checkpoints.append(
            Checkpoint(step=step, label=label, blob=self.capture_fn(env), base_xy=base_xy, phase=phase)
        )

    def nearest(self) -> Optional[Checkpoint]:
        """Most recent checkpoint (the one we rewind to)."""
        return self.checkpoints[-1] if self.checkpoints else None

    def observe(
        self,
        env,
        *,
        step: int,
        base_xy: tuple[float, float],
        yaw: float,
        holding: bool,
        dist_box: float,
        dist_goal: float,
        grasp_event: bool = False,
        drop_event: bool = False,
        out_of_bounds: bool = False,
        contact_force: float = 0.0,
    ) -> StepOutcome:
        """Update milestones + stall/progress trackers, then rewind to nearest if a trigger fires.

        Call once per control step AFTER env.step. Returns what happened (for logging).
        """
        out = StepOutcome()

        # Active target = box while approaching, final zone while carrying. Progress is measured
        # against THIS sub-goal; the baselines reset when the phase flips (grasp).
        phase = "B" if holding else "A"
        active_dist = dist_goal if holding else dist_box
        if phase != self._prog_phase:
            self._prog_phase = phase
            self._last_prog_dist = active_dist
            self._best_active = active_dist
            self._no_improve = 0
        if self._last_prog_dist is None:
            self._last_prog_dist = active_dist

        # No-progress tracker: an "improvement" is a new best (smallest) active distance.
        if self._best_active is None or active_dist < self._best_active - 1e-6:
            self._best_active = active_dist
            self._no_improve = 0
        else:
            self._no_improve += 1

        # ── 1a. milestone snapshots (each once) ──────────────────────────
        if not holding and not self._seen_approach and dist_box < self.approach_radius:
            self._seen_approach = True
            self._snapshot(env, step=step, label="approach_target", base_xy=base_xy, phase="A")
            out.snapshot_label = "approach_target"
        if grasp_event and not self._seen_grasp:
            self._seen_grasp = True
            self._snapshot(env, step=step, label="grasp", base_xy=base_xy, phase="B")
            out.snapshot_label = "grasp"
        if holding and not self._seen_delivery and dist_goal < self.approach_radius:
            self._seen_delivery = True
            self._snapshot(env, step=step, label="approach_delivery", base_xy=base_xy, phase="B")
            out.snapshot_label = "approach_delivery"

        # ── 1b. progress snapshot: only when the robot got materially CLOSER ──
        if out.snapshot_label is None and (self._last_prog_dist - active_dist) >= self.progress_delta:
            self._snapshot(env, step=step, label="progress", base_xy=base_xy, phase=phase)
            out.snapshot_label = "progress"

        # ── 1c. optional fixed-interval fallback (off by default) ────────
        if out.snapshot_label is None and self.period_steps and step > 0 and step % self.period_steps == 0:
            self._snapshot(env, step=step, label="periodic", base_xy=base_xy, phase=phase)
            out.snapshot_label = "periodic"

        # Advance the progress baseline whenever we took ANY snapshot (never on moving away).
        if out.snapshot_label is not None:
            self._last_prog_dist = active_dist

        # ── 2. update stall window ───────────────────────────────────────
        self._xy_hist.append(base_xy)
        self._yaw_hist.append(yaw)
        if self._cooldown > 0:
            self._cooldown -= 1

        # ── 3. triggers (immediate: collision/drop/out-of-bounds; windowed: spin/idle/no-progress) ──
        reason = self._trigger_reason(contact_force, drop_event, out_of_bounds)
        if reason and self._cooldown == 0 and self._rewinds < self.max_rewinds:
            target = self._escalated_target()
            if target is not None:
                self.restore_fn(env, target.blob)
                self._restore_counts[target.step] = self._restore_counts.get(target.step, 0) + 1
                self._rewinds += 1
                self._cooldown = self.cooldown_steps
                self._xy_hist.clear()
                self._yaw_hist.clear()
                self._no_improve = 0
                self._best_active = None
                out.restored = True
                out.reason = reason
                out.checkpoint_step = target.step
        return out

    def _escalated_target(self) -> Optional[Checkpoint]:
        """Nearest checkpoint, but drop it and fall back earlier if we keep failing at the same one.

        Once a checkpoint has been rewound to `escalate_after` times it is removed (we keep failing
        there), so the next rewind lands on an earlier, hopefully-recoverable state. 'start' (the only
        checkpoint when len==1) is never dropped.
        """
        target = self.nearest()
        while (target is not None and len(self.checkpoints) > 1
               and self._restore_counts.get(target.step, 0) >= self.escalate_after):
            self.checkpoints.pop()           # this unhelpful checkpoint is the most-recent one
            target = self.nearest()
        return target

    def _trigger_reason(self, contact_force: float, drop_event: bool, out_of_bounds: bool) -> Optional[str]:
        """Return the rewind reason, or None. Specific failures first, then windowed stall/no-progress."""
        if out_of_bounds:
            return "out_of_bounds"
        if drop_event:
            return "drop"                    # box dropped mid-carry -> rewind to where it was held
        if contact_force > self.collision_force_n:
            return "collision"
        if len(self._xy_hist) >= self.idle_steps:
            xs = [p[0] for p in self._xy_hist]
            ys = [p[1] for p in self._xy_hist]
            cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
            radius = max(math.hypot(x - cx, y - cy) for x, y in self._xy_hist)
            if radius < self.move_eps:       # parked in one spot
                yaw_travel = sum(abs(_angdiff(a, b))
                                 for a, b in zip(list(self._yaw_hist)[1:], self._yaw_hist))
                return "spinning" if yaw_travel > self.spin_yaw_threshold else "idle"
        if self._no_improve >= self.no_progress_steps:
            return "no_progress"             # moving, but not getting any closer
        return None


def _angdiff(a: float, b: float) -> float:
    """Smallest signed difference a-b wrapped to [-pi, pi]."""
    return (a - b + math.pi) % (2 * math.pi) - math.pi
