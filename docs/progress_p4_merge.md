# Merge Progress — branch `feat/p4-pickup-curriculum-fixes` → main

**Date:** 2026-06-21
**Author:** P4 (Kayla)
**Reviewed/approved:** Henry
**Scope:** P4 manipulation (pickup/curriculum) + P3 policy improvements + pipeline bug fixes.

Ringkasan semua yang diubah di branch ini, per komponen + per file.

---

## 1. P4 — Curriculum 4-stage (BARU, env API)
**File:** `env/curriculum.py`, `env/warehouse_env.py`, `tests/test_curriculum.py`
- `curriculum.py`: tambah `STAGE_NAV/GRASP/FULL/ANNEAL`, `validate_stage`, `stage_is_pregrasped`, `stage_is_spawn_near_box`, `resolve_goal_alpha`, `spawn_pose_near_box`. Unit-tested.
- `warehouse_env.py`: `WarehouseRLEnv.set_stage()` / `set_goal_alpha()`; per-stage reset (`_apply_stage_reset`): stage-1 pre-grasp (`_pregrasp_box`), stage-2 spawn-near-box (`_spawn_base_near_box`); `goal_position()` di-anneal pakai `goal_alpha`.
- **Mekanisme P4; transisi stage dipanggil P3 (lihat §5).**

## 2. P4 — Magnetic pickup + arm beku
**File:** `env/warehouse_env.py`, `env/grasp.py`
- Arm **dibekukan** di `WarehouseGymEnv.step` (EE action di-nol-kan) → robot gak reach/knock box.
- Grasp **by proximity, world-frame** (fix bug lama: dulu bandingin EE base-frame vs box env-local → gak pernah nyala).
- `GRIP_RADIUS_M` 0.10 → **0.25** (range "berhenti di depan", size-aware via `box_half`).
- Stage-2 standoff size-aware (heavy 0.55 / kecil 0.65).

## 3. P4 — Carry model
**File:** `env/warehouse_env.py`, `env/attach.py`
- Default `CARRY_MODE = "kinematic"` (hidden-kinematic): box ke-grab → **disembunyiin** (gak dirender, hemat compute) + **collision dimatiin** (gak nabrak rak) + diteleport ngikut robot (anchor di depan+atas chassis: `GRIP_FWD`/`GRIP_UP`); muncul + collision balik pas drop/reset.
- `attach.py` (BARU): physics `UsdPhysics.FixedJoint` weld `panda_hand`↔box — fallback `CARRY_MODE="physics"`.
- Release box hanya pas gripper buka (welded/kinematic carry gak bisa drift).

## 4. P4 — Stuck/idle timeout (BARU)
**File:** `env/warehouse_env.py`
- Term `stuck_timeout`: base diam (translate < `STUCK_MOVE_EPS_M`=0.02m/step) selama `STUCK_STEPS`=450 (~45s @10Hz) → reset. Bebasin robot mojok dinding tanpa nunggu 100s.
- **CATATAN:** episode cap tetap 100s (`episode_length_s`). 45s ini cuma kalau robot DIAM; kalau gerak terus, episode jalan penuh 100s.

## 5. P3 — Policy improvement (C1 + C2)
**File:** `policy/actor_critic.py`, `policy/train_loop.py`
- **C1 baseline:** actor loss kurangi critic value → advantage `(returns - V)` (turunin variance). `Actor.loss` terima arg `values` (optional, backward-compat).
- **C2 return normalization:** advantage dibagi EMA range return (5–95 percentile, DreamerV3). Buffer `ret_scale` persist di checkpoint.
- **Curriculum wiring:** `train_loop` panggil `env.set_stage()`/`set_goal_alpha()` via sliding success-rate gate (stage 1→3, anneal stage 4). Threshold dari cfg, no-op kalau env gak punya `set_stage`.

## 6. Bug fixes (pipeline)
**File:** `buffer/replay_buffer.py`, `tests/test_obs_adapter.py`, `training/logger.py`
- **HER infinite recursion** (`replay_buffer`): relabeled transition (done=True) micu `her_relabel` lagi → RecursionError. Fix: `_add_one(track=False)` buat relabeled add → gak re-track/re-trigger.
- **obs_adapter test basi**: update ke kontrak v2 (`goal_emb`→`goal_id` + manip keys).
- **logger wandb crash**: `wandb.init` gak login → uncaught → bunuh `P3Trainer.__init__` → Isaac teardown crash. Fix: honor `WANDB_MODE` env + try/except → fallback stdout.

## 7. Tooling / docs (BARU)
- `scripts/demo_pickup.py`: scripted pickup→carry→sort demo (no training).
- `scripts/tune_arm.py`: harness arm-reach + grasp-radius tuning.
- `docs/progress_p4.md`, `docs/tasks/p4_tasks.md`: status P4.

---

## Test status
Pure-logic suite (tanpa Isaac): **76 passed** (`pytest tests/ --ignore=tests/test_env.py --ignore=tests/test_obs.py`).
Termasuk 4 test yang sebelumnya fail (HER ×3 + obs_adapter) → sekarang ijo.

## ⚠️ BELUM diverifikasi di sim (perlu run di GPU box)
1. `train_p3` end-to-end dengan curriculum **start stage 1** (pre-grasp). Kalau reset crash → set `curriculum_start_stage=3` / `curriculum_enabled=False`.
2. Hidden-kinematic carry + collision-disable + stuck-timeout di sim asli.
3. Konstanta pickup (`GRIP_RADIUS_M`, standoff, `GRIP_FWD/UP`) belum dikalibrasi final.

Run: `$env:WANDB_MODE="offline"; python scripts/train_p3.py`
