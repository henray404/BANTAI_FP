# 2026-06-19 — DreamerV3 vanilla: migration fix valid, run blocked by Blackwell silent crash

## Context
First attempt to run training end-to-end after P1–P4 work. Chose the vanilla
NM512 DreamerV3 path (`scripts/train_dreamer.py`, config C1) — least code to a
real learning curve. The P3 path (`train_p3.py`) is NOT runnable: `WarehouseWorldModel`
(P2 deliverable) is not exported in `models/dreamerv3/__init__.py` (only abstract
`WorldModelInterface` exists), so non-`--mock_wm` raises ImportError.

## Part A — Migration fix (DONE, validated)
The vanilla path was stale: still nav contract (action `(2,)`, obs `goal_emb` 512).
Current env is pickup (action `(6,)`, obs `goal_id` one-hot + ee_pos/gripper/holding/box_pos).
Synced 3 files + test + dep:

- `models/dreamerv3/obs_adapter.py` — `VECTOR_KEYS` → pickup keys (drop `goal_emb`).
- `models/dreamerv3/warehouse_dreamer_env.py` — obs space dims pickup; `action_space` `(6,)`;
  `step` reshape `(6)`.
- `models/dreamerv3/config.py` — `encoder/decoder mlp_keys` → pickup regex; `time_limit` 600→1000;
  task `warehouse_pickup`.
- `tests/test_obs_adapter.py` — pickup keys. `pytest` → 3 passed.
- Installed `ruamel.yaml` into `isaaclab` (was missing; torch pin intact 2.7.0+cu128).

Validated WITHOUT Isaac: `build_config` OK, wrapper imports OK, mlp_keys correct.
Validated WITH Isaac: run #1 booted, completed prefill episodes (proves obs/action/
world-model wrapper wiring works end-to-end).

## Part B — Run blocked: Blackwell silent native crash (NOT code, NOT OOM)
Two runs on RTX 5050 Laptop (8GB, Blackwell sm_120, driver 580.88, headless):

| Run | Died at | metrics |
|---|---|---|
| #1 | during/after "Simulate agent" (past prefill=2000) | 2 prefill episodes: ep1 return -13.5 len 1000, ep2 return -0.5 len ~2 (out_of_bounds near north wall, expected for random policy) |
| #2 | sim init ~56s, BEFORE "Prefill dataset" | none (died at render init) |

### Signature → diagnosis
- Process returns to shell prompt with **NO Python Traceback** → native (C++/GPU) crash, not a Python exception.
- **Not OOM**: CUDA OOM always prints `CUDA out of memory` traceback. Absent. GPU freed to ~10 MiB after each death.
- **Non-deterministic crash point** (#1 training, #2 init) → hardware flakiness, not a fixed code/shape/OOM bug.
- Crash tied to render/camera init phase → matches `bugs_errors/2026-05-22_sdp-camera-crash-blackwell.md`.
  Driver pin 580.88 REDUCES but does not eliminate the Blackwell camera SDP crash.

## Resolution / next step
Code is correct (run #1 cleared prefill). Remaining failure is the Blackwell GPU, out of code scope.
**Move to non-Blackwell GPU (e.g. RTX 2080 Ti, Turing sm_75).** Turing avoids the camera SDP crash
class entirely; no driver pin needed. `precision: 32` in vendored config → no bf16 issue on Turing.
11GB/card is ample for batch 16×64 fp32 (one process uses one GPU; 2 GPUs ≠ pooled VRAM — use the
second card for a parallel seed instead).

## Notes
- Each failed Isaac run leaves a zombie `python.exe` (Blackwell `close()` hang) that does NOT hold VRAM.
  Kill between runs anyway.
- PowerShell `*>` / `Tee-Object` write UTF-16 logs; use `| Out-File -Encoding utf8` for greppable logs.
