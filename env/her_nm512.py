# env/her_nm512.py
# P5/P3 — Visual HER for the NM512 vendored DreamerV3 loop (configs #5, #6).
#
# The NM512 loop keeps a single in-memory episode cache (`train_eps`) that is BOTH
# mutated by tools.simulate AND sampled by make_dataset during training. So to make
# Visual HER episodes actually train the world model, the relabeled episode must be
# inserted into THAT cache (saving to disk alone only matters on reload).
#
# install_visual_her() monkeypatches two vendor functions (idempotent):
#   - tools.make_dataset : captures the train cache dict reference.
#   - tools.save_episodes: after a real episode is saved, builds a relabeled copy and
#                          inserts it into the captured cache (+ saves it to disk).
#
# Visual HER strategy (mirrors buffer/visual_her.py, adapted to the cache-of-arrays):
#   relabel only episodes where the box was grasped; set goal/goal_id to the zone the
#   robot got closest to after grasping; give success_reward + terminal on the last step.
#
# Pure numpy — no Isaac/torch. Unit-tested in tests/test_her_nm512.py.

"""Visual HER episode relabeling + monkeypatch installer for the NM512 DreamerV3 loop."""

from __future__ import annotations

import numpy as np

from buffer.visual_her import ZONE_GOAL_IDS, ZONE_POSITIONS

# Module-level state captured from the vendored loop by the monkeypatches.
_STATE: dict = {"cache": None, "rng": None, "ratio": 0.5, "reward": 10.0, "installed": False}


def _col(ep: dict, key: str) -> np.ndarray:
    """Return episode column `key` as a 2D array (T, d) for uniform indexing."""
    arr = np.asarray(ep[key], dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def relabel_cache_episode(
    ep: dict,
    zone_positions: np.ndarray = ZONE_POSITIONS,
    success_reward: float = 10.0,
) -> dict | None:
    """Build a Visual-HER-relabeled copy of one NM512 cache episode.

    Args:
        ep:             Cache episode dict (lists/arrays per key, length T over timesteps).
        zone_positions: (3,3) zone xyz, index == goal_id one-hot index.
        success_reward: Reward written on the final relabeled transition.

    Returns:
        A new episode dict with goal/goal_id relabeled to the achieved zone and a
        terminal success on the last step, or None if the box was never grasped
        (no positive signal to extract).
    """
    holding = _col(ep, "holding").reshape(-1)
    grasp_idx = next((i for i, h in enumerate(holding) if h >= 0.5), None)
    if grasp_idx is None:
        return None

    zone_pos = np.asarray(zone_positions, dtype=np.float32)
    position = _col(ep, "position")  # (T, 3)

    # Zone the robot got closest to after grasping.
    best_d, best_zone = np.inf, 0
    for t in range(grasp_idx, len(position)):
        for zi, zp in enumerate(zone_pos):
            d = float(np.linalg.norm(position[t, :2] - zp[:2]))
            if d < best_d:
                best_d, best_zone = d, zi

    new_goal = zone_pos[best_zone].astype(np.float32)
    new_goal_id = ZONE_GOAL_IDS[best_zone].astype(np.float32)

    out: dict = {}
    T = len(holding)
    for k, v in ep.items():
        arr = np.asarray(v)
        if k == "goal":
            out[k] = np.tile(new_goal, (T, 1)).astype(arr.dtype if arr.size else np.float32)
        elif k == "goal_id":
            out[k] = np.tile(new_goal_id, (T, 1)).astype(arr.dtype if arr.size else np.float32)
        else:
            out[k] = arr.copy()

    # Terminal success on the final transition.
    rew = np.asarray(out["reward"], dtype=np.float32).reshape(-1).copy()
    rew[-1] = float(success_reward)
    out["reward"] = rew
    if "is_terminal" in out:
        term = np.asarray(out["is_terminal"]).reshape(-1).copy()
        term[-1] = True
        out["is_terminal"] = term
    if "discount" in out:
        disc = np.asarray(out["discount"], dtype=np.float32).reshape(-1).copy()
        disc[-1] = 0.0
        out["discount"] = disc
    return out


def install_visual_her(her_ratio: float = 0.5, success_reward: float = 10.0,
                       seed: int = 0) -> None:
    """Monkeypatch the vendored NM512 tools so training samples Visual HER episodes.

    Call AFTER models.dreamerv3.config.add_vendor_to_path() and `import tools`, BEFORE
    dreamer.main(config). Idempotent.

    Args:
        her_ratio:      Fraction of grasped episodes to additionally relabel.
        success_reward: Reward on the relabeled terminal step.
        seed:           RNG seed for the stochastic relabel gate.
    """
    import tools  # vendored top-level module (on sys.path)

    _STATE.update(rng=np.random.default_rng(seed), ratio=float(her_ratio),
                  reward=float(success_reward))
    if _STATE["installed"]:
        return

    orig_make_dataset = tools.make_dataset
    orig_save_episodes = tools.save_episodes

    def make_dataset(episodes, config):
        """Capture the live train cache so HER episodes can be injected into it."""
        _STATE["cache"] = episodes
        return orig_make_dataset(episodes, config)

    def save_episodes(directory, episodes):
        """Save real episodes, then relabel + inject HER copies into the train cache."""
        ok = orig_save_episodes(directory, episodes)
        cache = _STATE["cache"]
        if cache is None:  # prefill phase: dataset not built yet, skip relabel
            return ok
        for ep_id, ep in list(episodes.items()):
            if _STATE["rng"].random() > _STATE["ratio"]:
                continue
            relabeled = relabel_cache_episode(ep, success_reward=_STATE["reward"])
            if relabeled is None:
                continue
            her_id = f"{ep_id}-her"
            cache[her_id] = relabeled
            orig_save_episodes(directory, {her_id: relabeled})
        return ok

    tools.make_dataset = make_dataset
    tools.save_episodes = save_episodes
    _STATE["installed"] = True
