# Run-all / ablation changes — 2026-06-23

What changed around `experiments/run_all.py` this session, and what every run now does
differently. `run_all.py` itself was **not** modified (its wiring was verified correct);
the changes are in the **config registry** it iterates and in the **env** every run uses.

---

## TL;DR
- Ablation trimmed **6 → 3 configs** (CA-SLOPE-only). `run_all` now schedules **9 runs** (was 18).
- `run_all.py` code unchanged — wiring verified end-to-end.
- Every run now trains on a new env: active arm (stage ≥2), grasp latch, failure resets +
  penalties, reset-to-checkpoint, and a reduced scene (8GB relief).

---

## 1. Ablation trimmed to CA-SLOPE-only — `experiments/configs.py`
**Removed:** `#1 sac`, `#5 dreamer_her`, `#6 dreamer_full`.
**Kept:** `#2 ppo`, `#3 dreamer_vanilla`, `#4 dreamer_caslope` (idx left at 2/3/4 so lognames
`c2_ppo` / `c3_dreamer_vanilla` / `c4_dreamer_caslope` and any prior results stay stable).

Reason: drop Visual HER entirely (#5, #6) and the SAC baseline (#1) — study CA-SLOPE only.

- `CONFIGS`: 6 → 3 entries.
- `ISOLATION_COMPARISONS`: 6 → 2 pairs → `(4,3)` "CA-SLOPE over world model", `(3,2)` "model-based vs PPO".
- `__main__` smoke assert: `6 configs / 18 runs` → `3 configs / 9 runs`.
- `by_idx` error message + module header/docstring updated.
- **`tests/test_experiments.py`** updated to match: `test_three_configs_nine_runs` (3 / 9 / idx {2,3,4})
  and `test_caslope_ablation_flags` (dreamer quad = `{(F,F),(T,F)}`, no HER anywhere).

Effect on `run_all`: schedule = 3 configs × 3 seeds = **9 runs** (`--only` / `--scenario` still
work by idx 2/3/4).

---

## 2. `run_all.py` — verified, NOT changed
Wiring confirmed correct:
- `train_dreamer.py` flags used (`--seed --steps --logdir --headless --config --ca_slope --visual_her`) — all exist.
- `train_sac.py` flags used (`--algo --seed --timesteps --logdir --headless --config --ca_slope`) — all exist.
- `experiments.configs` (`CONFIGS`, `ExperimentConfig.idx/algo/ca_slope/visual_her/logname`) + `experiments.settings.load_settings` (`budget["seeds"]`, `["total_steps"]`) — all present.
- Resumable `DONE` markers, `--only/--seeds/--device/--timeout/--dry-run` all intact.

**Caveats (flags, not bugs):**
- `--stage` is **never passed** → every run uses the default **stage 3 (full chain)**. There is no
  stage-2 (grasp-focus) sweep unless a `--stage` passthrough is added.
- 8GB GPUs: each run spins Isaac + camera. Full scene OOMs (BAR1) → see §4 scene knobs.

---

## 3. Env behavior every run now trains on
(`env/warehouse_env.py`, `warehouse_reward.py`, `reward_pickup.py`, `warehouse_scene.py`)

| Area | Change |
|---|---|
| Arm | **Active arm for curriculum stage ≥2** (absolute-hold EE target + DLS pose IK + clamp stack, ported from `drive_env_v2`); frozen at stage 1. |
| Grasp | **Latched ("nyantol")**: once grasped the box stays FixedJoint-welded; only a weld-break safety drops it. Carry freezes the arm at the grab pose. |
| Terminations | **crash (>50 N), dropped box, no-grasp 30 s timeout (stage 2)** → episode reset. (plus existing success / out-of-bounds / stuck / time-out) |
| Penalties | **`failure` −15** one-shot on any failure reset; **`under_rack` −2/step**; **`carry_regress` −0.1/step** when a held box backs up / dawdles toward the zone. |
| Reset-to-checkpoint | **(num_envs=1)** snapshot at grasp + each closer 2 m ring; a failure reset resumes from the last checkpoint (clean `done`, not mid-episode rewind). Fail-safe → fresh spawn if restore throws. |
| Logging | per-step weighted reward breakdown surfaced in `step()` `info["reward_terms"]` / `info["reward_total"]`. |

---

## 4. Tunables (no code edit needed)
**`configs/reward_weights.yaml`** — added: `failure: -15.0`, `under_rack: 2.0`, `carry_regress: 0.1`
(plus existing approach/grasp/carry/deliver/time_pen/collision/drop/idle*).

**`configs/env_config.yaml` → `scene:`** (8 GB VRAM relief; read by `warehouse_scene._scene_knobs`):
- `num_boxes: 3` — spawn only N boxes **and the matching racks+decks** (box i ↔ rack i): 3 → 3 racks
  + 9 decks + 3 boxes instead of 18 + 54 + 18. `0` = full warehouse. Multiples of 3.
- `spawn_props: false` — skip the 8 decorative props.

Thresholds as code constants in `warehouse_env.py` / `warehouse_reward.py`:
`COLLIDE_RESET_N=50`, `NO_GRASP_TIMEOUT_STEPS=300`, `CARRY_REGRESS_STEPS=50`, `CHECKPOINT_RING_M=2.0`.

---

## 5. How to run
```bash
conda activate isaaclab

# dry-run: print the 9-run schedule
python -m experiments.run_all --dry-run

# smoke: all 9, tiny budget
python -m experiments.run_all --steps 5000

# full ablation: 9 runs sequential, headless, 200k steps each
python -m experiments.run_all

# aggregate + significance
python -m experiments.analyze --results training/results/ablation
```
Useful: `--only 4`, `--seeds 0`, `--device 0/1`, `--timeout <sec>`. Blackwell: kill the zombie
`python.exe` between runs.

---

## 6. Known blockers (why a full local `run_all` may not finish)
- **RTX 5050 (8 GB, Windows)**: full scene → BAR1 OOM → PhysX CUDA-700. Use `num_boxes:3` + headless
  (already set). Camera works on driver 580.88. → `bugs_errors/2026-06-23_oom-bar1-physx-error700.md`.
- **2× RTX 2080 Ti (Manjaro lab box, driver 595.71.05)**: Isaac RTX renderer can't init headless
  (`GLFW initialization failed` → `omni.hydra.rtx` segfault, `rc=-11` at `newStage`). Every run dies
  at startup — environment/driver, not project code. Needs an Isaac-5.1-supported driver.
- **Colab A100**: Isaac Sim does **not** run on Colab. `run_all` / `collect_offline` / `train_dreamer`
  are local-only. Colab runs **`scripts/train_offline.py`** (no sim) on pre-collected `.npz` episodes —
  collect locally on a working Isaac box, upload, then train on A100 (see `notebook.ipynb`).

---

## Files touched this session (committed)
`experiments/configs.py`, `tests/test_experiments.py`, `env/warehouse_env.py`,
`env/warehouse_reward.py`, `env/reward_pickup.py`, `env/reward_debug.py`,
`env/warehouse_scene.py`, `configs/env_config.yaml`, `configs/reward_weights.yaml`,
`scripts/drive_env.py`, bug docs. (`experiments/run_all.py` itself: unchanged.)
