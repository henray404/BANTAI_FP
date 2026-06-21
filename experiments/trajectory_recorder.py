# experiments/trajectory_recorder.py
# P5 — record the BEST eval episode's per-step decisions so it can be replayed in the
# Isaac Lab GUI (see the best run visually, without retraining or reloading the agent).
#
# What it captures, per the user's request:
#   - best_trajectory.csv : one row per step — the model's action (6,) + key state.
#   - best_init.json      : the scene snapshot at episode start (robot + box poses, goal),
#                           so the GUI replay restores the SAME scene before applying actions
#                           (action-only replay would diverge — boxes/goal are randomized).
#
# "Best" = highest eval success, ties broken by episode return. Overwritten when beaten.
# scripts/replay_best.py consumes both files.
#
# PURE python (no Isaac/torch): the caller passes plain floats/lists. The Isaac side
# (snapshot dict + reading state) lives in env/scene_snapshot.py. Unit-tested.

"""Record + persist the best eval episode's action trace for GUI replay."""

from __future__ import annotations

import csv
import json
from pathlib import Path

# CSV columns: step, the 6 action dims, robot base xyz, ee xyz, holding, reward.
TRAJ_HEADER = (
    "step", "a0", "a1", "a2", "a3", "a4", "a5",
    "robot_x", "robot_y", "robot_z", "ee_x", "ee_y", "ee_z", "holding", "reward",
)


class TrajectoryRecorder:
    """Buffer the current episode; persist it iff it beats the best seen so far.

    Usage per eval episode:
        rec.begin(init_snapshot)                 # at reset
        rec.step(i, action, robot_xyz, ee_xyz, holding, reward)   # each step
        rec.end(success_rate=1.0, ep_return=12.3)                  # at done -> maybe save
    """

    def __init__(self, outdir: str | Path,
                 csv_name: str = "best_trajectory.csv",
                 init_name: str = "best_init.json"):
        """Persist into `outdir` (typically <logdir>/best)."""
        self.dir = Path(outdir)
        self._csv = self.dir / csv_name
        self._init = self.dir / init_name
        self._best_key = (float("-inf"), float("-inf"))  # (success, return)
        self._rows: list[list] = []
        self._snapshot: dict | None = None

    def begin(self, init_snapshot: dict | None) -> None:
        """Start a new episode: clear the row buffer, stash the scene snapshot."""
        self._rows = []
        self._snapshot = init_snapshot

    def step(self, step_idx: int, action, robot_xyz, ee_xyz,
             holding, reward) -> None:
        """Append one step's action + key state to the episode buffer."""
        a = [float(x) for x in list(action)[:6]]
        a += [0.0] * (6 - len(a))
        rx = [float(x) for x in list(robot_xyz)[:3]]
        ex = [float(x) for x in list(ee_xyz)[:3]]
        self._rows.append([int(step_idx), *a, *rx, *ex,
                           float(holding), float(reward)])

    def end(self, success_rate: float, ep_return: float) -> bool:
        """If this episode beats the best (success, then return), persist it. Returns saved?."""
        key = (float(success_rate), float(ep_return))
        if key <= self._best_key or not self._rows:
            return False
        self._best_key = key
        self.dir.mkdir(parents=True, exist_ok=True)
        with self._csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(TRAJ_HEADER)
            w.writerows(self._rows)
        meta = {"success_rate": key[0], "ep_return": key[1],
                "steps": len(self._rows), "init": self._snapshot}
        self._init.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return True
