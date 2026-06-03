# test_env.py — Automated sanity check for the warehouse env (interface contract).
#
# Usage:
#   python tests/test_env.py --num_envs 1
#
# Prints PASS/FAIL for each checklist item. Exit status nonzero on any FAIL.

"""Verify WarehouseGymEnv matches the team's interface contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Warehouse env automated tests")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # warehouse env always uses onboard camera

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from gymnasium import spaces  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from env.warehouse_env import GOAL_EMB_DIM, IMG_HW, WarehouseEnvCfg, WarehouseGymEnv  # noqa: E402


Result = tuple[str, bool, str]


def _record(results: list[Result], name: str, ok: bool, note: str = "") -> None:
    """Append a check result tuple."""
    results.append((name, ok, note))


def run_tests(num_envs: int) -> list[Result]:
    """Run full checklist; return list of (name, passed, note)."""
    results: list[Result] = []

    try:
        cfg = WarehouseEnvCfg()
        cfg.scene.num_envs = num_envs
        _record(results, "WarehouseEnvCfg instantiates", True)
    except Exception as e:
        _record(results, "WarehouseEnvCfg instantiates", False, repr(e))
        return results

    try:
        env = WarehouseGymEnv(cfg=cfg)
        obs, _ = env.reset()
        _record(results, "env.reset() returns dict", isinstance(obs, dict),
                f"keys={sorted(obs.keys())}")
    except Exception as e:
        _record(results, "env.reset() returns dict", False, repr(e))
        return results

    expected_shapes = {
        "pixels":   (num_envs, 3, IMG_HW, IMG_HW),
        "position": (num_envs, 3),
        "goal":     (num_envs, 3),
        "goal_emb": (num_envs, GOAL_EMB_DIM),
        "heading":  (num_envs, 2),
    }
    for key, want in expected_shapes.items():
        try:
            got = tuple(obs[key].shape)
            _record(results, f"obs['{key}'].shape == {want}", got == want, f"got={got}")
        except Exception as e:
            _record(results, f"obs['{key}'].shape", False, repr(e))

    try:
        a = env.action_space
        ok = (
            isinstance(a, spaces.Box)
            and a.shape == (2,)
            and float(a.low.min()) == -1.0
            and float(a.high.max()) == 1.0
        )
        _record(results, "action_space == Box(-1, 1, shape=(2,))", ok, str(a))
    except Exception as e:
        _record(results, "action_space check", False, repr(e))

    try:
        for _ in range(10):
            action = np.random.uniform(-1.0, 1.0, size=(num_envs, 2)).astype(np.float32)
            obs, reward, term, trunc, _ = env.step(action)
        _record(results, "env.step() runs 10 steps", True)
        
        # Verify box physics (boxes shouldn't fall through shelves)
        from env.warehouse_scene import ITEM_SPECS
        fallen = 0
        for name, _, _, _ in ITEM_SPECS:
            box = env._env.scene[name]
            z = box.data.root_pos_w[0, 2].item()
            if z < 0.5:
                fallen += 1
        _record(results, "Physics: Boxes remain on shelves (z > 0.5)", fallen == 0, f"Fallen: {fallen}")
            
    except Exception as e:
        _record(results, "env.step() / physics check", False, repr(e))
        env.close()
        return results

    try:
        frame = env.render()
        ok = (
            isinstance(frame, np.ndarray)
            and frame.shape == (IMG_HW, IMG_HW, 3)
            and frame.dtype == np.uint8
        )
        _record(results, f"env.render() -> uint8 ({IMG_HW},{IMG_HW},3)", ok,
                f"shape={frame.shape} dtype={frame.dtype}")
    except Exception as e:
        _record(results, "env.render() returns frame", False, repr(e))

    try:
        ok = isinstance(reward, torch.Tensor) and reward.shape == (num_envs,)
        _record(results, "reward shape == (num_envs,)", ok, f"got={tuple(reward.shape)}")
    except Exception as e:
        _record(results, "reward shape check", False, repr(e))

    try:
        env.close()
        _record(results, "env.close() runs cleanly", True)
    except Exception as e:
        _record(results, "env.close() runs cleanly", False, repr(e))

    return results


def print_summary(results: list[Result]) -> bool:
    """Print PASS/FAIL summary; return True iff every check passed."""
    print("\n=== Warehouse Env Test ===")
    all_pass = True
    for name, ok, note in results:
        status = "PASS" if ok else "FAIL"
        suffix = f" — {note}" if note else ""
        print(f"  [{status}] {name}{suffix}")
        if not ok:
            all_pass = False
    print("\n=== ALL PASS ===" if all_pass else "\n=== SOME FAILED ===")
    return all_pass


if __name__ == "__main__":
    results = run_tests(args_cli.num_envs)
    ok = print_summary(results)
    simulation_app.close()
    sys.exit(0 if ok else 1)
