# experiments/nm512_eval.py
# P5 — record success-rate eval rows from the NM512 DreamerV3 eval env.
#
# NM512's native logger reports eval_return/eval_length but not task success rate. This
# thin wrapper sits at the OUTERMOST eval-env level (forwarding the NM512 env API via
# __getattr__) and, every `eval_episodes` completed episodes, flushes one row to the
# shared eval_metrics.csv — so DreamerV3 runs produce the SAME metric file as the SAC/PPO
# baselines and analyze.py can compare all six configs uniformly.
#
# Eval cadence in NM512 is fixed (config.eval_every), and the eval env is built once and
# reused across all eval phases. So the k-th flush corresponds to env step ~ k*eval_every.

"""Outer eval-env wrapper that logs DreamerV3 success rate to eval_metrics.csv."""

from __future__ import annotations

import numpy as np

from experiments.metrics import BestModelTracker, EvalCsv, _rl_env, episode_success


class EvalRecorder:
    """Wrap an NM512 eval env; flush success/length to a CSV once per eval phase.

    Args:
        nm512_env:     Fully-built NM512 eval env (output of make_warehouse_dreamer).
        warehouse_env: The underlying WarehouseGymEnv (for success readout).
        csv:           experiments.metrics.EvalCsv to append rows to.
        eval_episodes: Episodes per eval phase (one CSV row per phase).
        eval_every:    Env steps between eval phases (for the row's step column).
        best:          Optional BestModelTracker to snapshot the best checkpoint.
        checkpoint_src: Path to the live checkpoint (NM512 writes <logdir>/latest.pt).
    """

    def __init__(self, nm512_env, warehouse_env, csv: EvalCsv,
                 eval_episodes: int = 5, eval_every: int = 10_000,
                 best: BestModelTracker | None = None, checkpoint_src=None,
                 traj=None):
        """Hold the wrapped env + CSV writer and reset counters."""
        self._env = nm512_env
        self._warehouse = warehouse_env
        self._csv = csv
        self._best = best
        self._checkpoint_src = checkpoint_src
        self._traj = traj  # TrajectoryRecorder for the best-episode action trace (or None)
        self._eval_episodes = int(eval_episodes)
        self._eval_every = int(eval_every)
        self._succ: list[float] = []
        self._lens: list[float] = []
        self._rets: list[float] = []
        self._ep_len = 0
        self._ep_ret = 0.0
        self._phase = 0

    def __getattr__(self, name):
        """Forward unknown attributes (id, observation_space, ...) to the wrapped env."""
        return getattr(self._env, name)

    def reset(self, *args, **kwargs):
        """Reset the wrapped env, the per-episode accumulators, and the trajectory buffer."""
        self._ep_len = 0
        self._ep_ret = 0.0
        out = self._env.reset(*args, **kwargs)
        if self._traj is not None:
            from env.scene_snapshot import capture_init_state
            self._traj.begin(capture_init_state(_rl_env(self._warehouse)))
        return out

    def step(self, action):
        """Step the wrapped env; record the action trace; on done flush + save best."""
        obs, reward, done, info = self._env.step(action)
        self._ep_len += 1
        self._ep_ret += float(reward)
        if self._traj is not None:
            from env.scene_snapshot import read_replay_state
            a = action["action"] if isinstance(action, dict) else action
            robot_xyz, ee_xyz, holding = read_replay_state(_rl_env(self._warehouse))
            self._traj.step(self._ep_len, a, robot_xyz, ee_xyz, holding, float(reward))
        if done:
            succ = 1.0 if episode_success(self._warehouse) else 0.0
            self._succ.append(succ)
            self._lens.append(self._ep_len)
            self._rets.append(self._ep_ret)
            if self._traj is not None:
                self._traj.end(succ, self._ep_ret)
            self._ep_len = 0
            self._ep_ret = 0.0
            if len(self._succ) >= self._eval_episodes:
                self._flush()
        return obs, reward, done, info

    def _flush(self) -> None:
        """Write one averaged eval row, snapshot the best model, reset accumulators."""
        n = len(self._succ)
        step = self._phase = self._phase + 1
        step *= self._eval_every
        metrics = {
            "success_rate": sum(self._succ) / n,
            "success_std": float(np.std(self._succ)),
            "mean_length": sum(self._lens) / n,
            "mean_return": sum(self._rets) / n,
        }
        self._csv.log(step, metrics)
        if self._best is not None:
            self._best.update(step, metrics, checkpoint_src=self._checkpoint_src)
        self._succ.clear()
        self._lens.clear()
        self._rets.clear()
