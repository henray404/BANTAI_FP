# experiments/eval_harness.py
# Person 5 — headless eval harness. Runs scenarios x seeds, applies CA-SLOPE shaping each step,
# and records (a) a per-STEP trace CSV (rekam jejak tiap langkah robot) and (b) a per-EPISODE
# summary CSV (performa per skenario). Pure-numpy + stdlib csv — runs on a Mac with no Isaac.
#
# RQ2 ablation modes (--mode):
#   category : CA-SLOPE, per-category gains   (the proposed method)
#   generic  : SLOPE, one gain for all        (the control)
#   none     : no shaping (slope_reward = 0)  (vanilla base reward)
#
# Plug the real stack later: pass a different env factory (any reset/step matching ToyPickupEnv) and
# a learned policy(obs)->action(6,); the CSV/metrics path is unchanged.

"""Run pickup scenarios headless, apply CA-SLOPE, write per-step + per-episode CSV traces."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments.scenarios import DEFAULT_SCENARIOS, Scenario
from experiments.scripted_policy import ScriptedPickupPolicy
from experiments.toy_pickup_env import ToyPickupEnv
from reward.ca_slope import CASlopeShaper, CATEGORY_NAMES

# Per-step trace columns. One row per robot step — the rekam jejak.
STEP_COLUMNS = [
    "run_id", "mode", "scenario", "category", "color", "seed", "step", "phase",
    "holding", "gripper",
    "base_x", "base_y", "base_yaw",
    "ee_x", "ee_y", "ee_z", "box_x", "box_y", "box_z", "goal_x", "goal_y",
    "dist_ee_box", "dist_box_goal",
    "a_base_lin", "a_base_ang", "a_ee_dx", "a_ee_dy", "a_ee_dz", "a_grip",
    "base_reward", "slope_reward", "total_reward", "cum_base_return", "cum_total_return",
    "grasp_event", "deliver_event", "drop_event", "done", "success",
]

# Per-episode summary columns. One row per scenario x seed — the performa.
SUMMARY_COLUMNS = [
    "run_id", "mode", "scenario", "category", "color", "seed",
    "steps", "success", "grasp_step", "deliver_step",
    "return_base", "return_total", "return_slope", "final_dist_box_goal",
]


def _state_dict(obs: dict) -> dict:
    """Pull the five CA-SLOPE state arrays out of an obs dict (batch them to shape (1, ...))."""
    return {
        "ee_pos": obs["ee_pos"][None, :],
        "box_pos": obs["box_pos"][None, :],
        "goal_pos": obs["goal_pos"][None, :],
        "holding": np.array([obs["holding"]], dtype=np.float64),
        "goal_id": obs["goal_id"][None, :],
    }


@dataclass
class EpisodeResult:
    """Aggregated per-episode metrics (mirrors one SUMMARY_COLUMNS row)."""

    scenario: str
    category: int
    color: str
    seed: int
    steps: int
    success: bool
    grasp_step: int
    deliver_step: int
    return_base: float
    return_total: float
    return_slope: float
    final_dist_box_goal: float


class EvalHarness:
    """Drive scenarios with a policy, shape with CA-SLOPE, stream rows into two CSV writers."""

    def __init__(
        self,
        mode: str = "category",
        scenarios: list[Scenario] | None = None,
        seeds: tuple[int, ...] = (0, 1, 2),
        max_steps: int = 600,
        shaper: CASlopeShaper | None = None,
        policy=None,
        run_id: str = "toy",
    ):
        """mode in {category, generic, none}; seeds = the 3-seed protocol; policy is callable(obs)->action."""
        if mode not in ("category", "generic", "none"):
            raise ValueError(f"mode must be category|generic|none, got {mode!r}")
        self.mode = mode
        self.scenarios = scenarios if scenarios is not None else DEFAULT_SCENARIOS
        self.seeds = seeds
        self.max_steps = max_steps
        self.run_id = run_id
        if shaper is None:
            shaper = CASlopeShaper(category_aware=(mode == "category"))
        self.shaper = shaper
        self.policy = policy if policy is not None else ScriptedPickupPolicy()

    def _slope(self, prev_obs: dict, next_obs: dict, done: bool) -> float:
        """CA-SLOPE shaping for one transition (0.0 when mode == none)."""
        if self.mode == "none":
            return 0.0
        f = self.shaper.shaping(
            _state_dict(prev_obs), _state_dict(next_obs), done=np.array([done], dtype=np.float64)
        )
        return float(np.asarray(f).reshape(-1)[0])

    def run_episode(self, scenario: Scenario, seed: int, step_writer=None, recorder=None) -> EpisodeResult:
        """Run one scenario/seed to termination, streaming step rows; return the episode summary.

        If `recorder` (a TrajectoryRecorder) is given, each step is also recorded as its own run and
        the run-level summary (success/steps/deliver_step/return) is set for per-scenario ranking.
        """
        env = ToyPickupEnv(max_steps=self.max_steps)
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        obs, _ = env.reset(scenario, seed=seed)

        cum_base = cum_total = cum_slope = 0.0
        grasp_step = deliver_step = -1
        success = False
        step = 0

        while True:
            action = np.asarray(self.policy(obs), dtype=np.float64).reshape(6)
            next_obs, base_r, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            slope_r = self._slope(obs, next_obs, done)
            total_r = base_r + slope_r
            cum_base += base_r
            cum_total += total_r
            cum_slope += slope_r
            step += 1
            if info["grasp_event"] and grasp_step < 0:
                grasp_step = step
            if info["deliver_event"] and deliver_step < 0:
                deliver_step = step
            success = success or bool(info["success"])

            if step_writer is not None or recorder is not None:
                row = self._step_row(scenario, seed, step, info, next_obs, action,
                                     base_r, slope_r, total_r, cum_base, cum_total, done)
                if step_writer is not None:
                    step_writer.writerow(row)
                if recorder is not None:
                    recorder.add(row)

            obs = next_obs
            if done:
                break

        if recorder is not None:
            recorder.set_summary({"success": int(success), "steps": step,
                                  "deliver_step": deliver_step, "grasp_step": grasp_step,
                                  "return": round(cum_total, 4)})

        final_dist = float(np.linalg.norm(obs["box_pos"][:2] - obs["goal_pos"][:2]))
        return EpisodeResult(
            scenario=scenario.name, category=scenario.category, color=scenario.color, seed=seed,
            steps=step, success=success, grasp_step=grasp_step, deliver_step=deliver_step,
            return_base=cum_base, return_total=cum_total, return_slope=cum_slope,
            final_dist_box_goal=final_dist,
        )

    def _step_row(self, scn, seed, step, info, obs, action, base_r, slope_r, total_r,
                  cum_base, cum_total, done) -> dict:
        """Assemble one STEP_COLUMNS row from the post-step state."""
        ee, box, goal = obs["ee_pos"], obs["box_pos"], obs["goal_pos"]
        return {
            "run_id": self.run_id, "mode": self.mode, "scenario": scn.name,
            "category": CATEGORY_NAMES[scn.category], "color": scn.color, "seed": seed,
            "step": step, "phase": info["phase"], "holding": int(obs["holding"]),
            "gripper": round(obs["gripper"], 3),
            "base_x": round(float(obs["position"][0]), 4), "base_y": round(float(obs["position"][1]), 4),
            "base_yaw": round(float(obs["heading_yaw"]), 4),
            "ee_x": round(float(ee[0]), 4), "ee_y": round(float(ee[1]), 4), "ee_z": round(float(ee[2]), 4),
            "box_x": round(float(box[0]), 4), "box_y": round(float(box[1]), 4), "box_z": round(float(box[2]), 4),
            "goal_x": round(float(goal[0]), 4), "goal_y": round(float(goal[1]), 4),
            "dist_ee_box": round(float(np.linalg.norm(ee - box)), 4),
            "dist_box_goal": round(float(np.linalg.norm(box[:2] - goal[:2])), 4),
            "a_base_lin": round(float(action[0]), 4), "a_base_ang": round(float(action[1]), 4),
            "a_ee_dx": round(float(action[2]), 4), "a_ee_dy": round(float(action[3]), 4),
            "a_ee_dz": round(float(action[4]), 4), "a_grip": round(float(action[5]), 4),
            "base_reward": round(base_r, 4), "slope_reward": round(slope_r, 4),
            "total_reward": round(total_r, 4), "cum_base_return": round(cum_base, 4),
            "cum_total_return": round(cum_total, 4),
            "grasp_event": int(info["grasp_event"]), "deliver_event": int(info["deliver_event"]),
            "drop_event": int(info["drop_event"]), "done": int(done), "success": int(info["success"]),
        }

    def run(self, out_dir: str | Path, record_dir: str | Path | None = None) -> list[EpisodeResult]:
        """Run every scenario x seed; write steps_/summary_ CSVs.

        If `record_dir` is given, ALSO record each episode as its own replayable run
        (<record_dir>/<scenario>_<mode>_seed<seed>.csv + .meta.json) so scripts/rank_runs.py can
        score them per scenario and pick the best demo run.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        steps_path = out / f"steps_{self.mode}.csv"
        summary_path = out / f"summary_{self.mode}.csv"

        results: list[EpisodeResult] = []
        with open(steps_path, "w", newline="") as sf, open(summary_path, "w", newline="") as mf:
            step_writer = csv.DictWriter(sf, fieldnames=STEP_COLUMNS)
            step_writer.writeheader()
            summary_writer = csv.DictWriter(mf, fieldnames=SUMMARY_COLUMNS)
            summary_writer.writeheader()
            for scn in self.scenarios:
                for seed in self.seeds:
                    recorder = self._make_recorder(record_dir, scn, seed) if record_dir else None
                    res = self.run_episode(scn, seed, step_writer=step_writer, recorder=recorder)
                    if recorder is not None:
                        recorder.close()
                    results.append(res)
                    summary_writer.writerow(self._summary_row(res))
        return results

    def _make_recorder(self, record_dir, scn: Scenario, seed: int):
        """Create a per-episode TrajectoryRecorder tagged with the scenario for ranking."""
        from recording.recorder import TrajectoryRecorder
        stem = Path(record_dir) / f"{scn.name}_{self.mode}_seed{seed}"
        meta = {
            "run_id": stem.name, "policy": "scripted_toy", "seed": seed,
            "scenario": scn.name, "category": CATEGORY_NAMES[scn.category], "color": scn.color,
            "slope_mode": self.mode, "control_hz": 10.0, "source": "toy_eval_harness",
        }
        return TrajectoryRecorder(stem, metadata=meta)

    def _summary_row(self, r: EpisodeResult) -> dict:
        """One SUMMARY_COLUMNS row from an EpisodeResult."""
        return {
            "run_id": self.run_id, "mode": self.mode, "scenario": r.scenario,
            "category": CATEGORY_NAMES[r.category], "color": r.color, "seed": r.seed,
            "steps": r.steps, "success": int(r.success), "grasp_step": r.grasp_step,
            "deliver_step": r.deliver_step, "return_base": round(r.return_base, 4),
            "return_total": round(r.return_total, 4), "return_slope": round(r.return_slope, 4),
            "final_dist_box_goal": round(r.final_dist_box_goal, 4),
        }
