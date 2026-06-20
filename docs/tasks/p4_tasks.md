# P4 — Manipulation: Tugas & Status

**Owner:** P4 (pairs w/ P1)
**Scope (spec §6):** grasp detection · pick-place curriculum · EE control tuning
**Date:** 2026-06-16
**Source:** dibaca langsung dari codebase (`env/grasp.py`, `env/reward_pickup.py`, `env/action_pickup.py`, `env/curriculum.py`, `env/warehouse_env.py`)
**Ref:** `docs/superpowers/specs/2026-06-08-pure-dl-pickup-redesign.md` §4, §7, §9

---

## Ringkasan status

| Area | Status | Catatan |
|---|---|---|
| Grasp detection | ✅ Done | surface-contact size-aware, unit-tested |
| Staged reward (grasp/carry/deliver/drop) | ✅ Done | `reward_pickup.py`, unit-tested |
| EE action split/scale | ✅ Done | `action_pickup.py`, unit-tested |
| IK wiring (DifferentialIK) | ✅ Done | `ActionsCfg.arm_ik` (P1 pair) |
| Grasp loop integrasi | ✅ Done | `WarehouseRLEnv.update_grasp()` |
| **Curriculum 4-stage** | ✅ Done | env API + per-stage reset + anneal, unit-tested (`curriculum.py`, `warehouse_env.py`); scheduler diserahkan ke P3/P5 |
| **EE control tuning** | ⚠️ Harness siap, run pending | `scripts/tune_arm.py` — jalankan di sim, paste hasil |
| **Arm reach end-to-end** | ⚠️ Harness siap, run pending | diverify oleh `scripts/tune_arm.py` (same run) |
| Carry model | ✅ Diputuskan: physics grasp | FixedJoint weld `panda_hand`↔box (`env/attach.py`), `CARRY_MODE`; kinematic = fallback. **Perlu sim-verify** (GPU PhysX runtime joint) |

> **Update 2026-06-20:** curriculum stage manager + physics grasp dibangun (auto mode).
> Tersisa SIM-DEPENDENT (butuh Isaac Sim di GPU box, tak bisa headless dari agent):
> 1. jalankan `python scripts/tune_arm.py` → verify arm reach + kalibrasi `EE_STEP_M`/`GRIP_RADIUS_M`.
> 2. verify physics-grasp weld bertahan di GPU pipeline (box ikut EE, tak jatuh); kalau gagal set `CARRY_MODE="kinematic"` di `env/warehouse_env.py`.
> 3. verify Stage-2 spawn-near-box (`_spawn_base_near_box`) drop chassis di samping box.

---

## ✅ Sudah selesai

### 1. Grasp detection — `env/grasp.py`
Model **proximity-to-surface** (bukan enclosure/lift) karena box (0.21/0.32/0.52 m) lebih besar dari bukaan gripper Franka.
- `grasp_success(ee_pos, box_pos, gripper_closed, box_half)` → `(jarak_EE_ke_permukaan < GRIP_RADIUS_M) AND gripper_closed`
- `grasp_lost(...)` → holding tapi EE pisah > 2×radius dari permukaan
- `GRIP_RADIUS_M = 0.10`
- Test: `tests/test_grasp.py` (5 test, termasuk surface-grasp box besar)

### 2. Staged reward — `env/reward_pickup.py`
- `approach_box_distance` — Phase A dense, jarak(ee,box), nol saat holding
- `carry_distance` — Phase B dense, jarak xy(box,zona), nol saat tidak holding
- `grasp_success_reward` — +1 one-shot saat grasp
- `drop_penalty` — +1 one-shot saat box jatuh di luar zona
- `pickup_delivered` / `pickup_delivered_reward` — holding AND box dalam radius 1.5 m zona
- Bobot di `RewardsCfg` (`warehouse_env.py`): approach -0.01, grasp +5, carry -0.01, deliver +10, time -0.005, collision -5, drop -2
- Test: `tests/test_reward_pickup.py`

### 3. EE action — `env/action_pickup.py`
- `split_action((N,6))` → `(base2, ee3, grip1)`; `ee3 = action[:,2:5] * EE_STEP_M`
- `EE_STEP_M = 0.05` m per step @ action=1.0
- Test: `tests/test_action_pickup.py`

### 4. IK wiring — `ActionsCfg.arm_ik` di `warehouse_env.py`
- `DifferentialInverseKinematicsActionCfg`, `command_type="position"`, `use_relative_mode=True`, `ik_method="dls"`, body `panda_hand`
- EE delta base-frame (lokal); orientasi top-down dikunci controller

### 5. Grasp loop — `WarehouseRLEnv.update_grasp()`
Dipanggil `WarehouseGymEnv.step` tiap step. Set `grasp_event`/`drop_event`/`holding`, lalu `_carry_held_boxes` (kinematik).

---

## ✅ Curriculum 4-stage (selesai — env API)

**Dibangun 2026-06-20.** Helper pure di `env/curriculum.py` + wiring sim di `env/warehouse_env.py`:
- `STAGE_NAV/GRASP/FULL/ANNEAL`, `validate_stage`, `stage_is_pregrasped`, `stage_is_spawn_near_box`, `resolve_goal_alpha`, `spawn_pose_near_box` — unit-tested (`tests/test_curriculum.py`).
- `WarehouseRLEnv.set_stage(n)` / `set_goal_alpha(a)` — API untuk P3/P5.
- `goal_position()` kali `goal_alpha` (anneal_goal) — default 1.0 = perilaku lama.
- `_apply_stage_reset`: stage 1 → `_pregrasp_box` (box di-snap ke EE + weld + holding=True); stage 2 → `_spawn_base_near_box` (chassis ditaruh di samping box).
- **P4 sediakan mekanisme, BUKAN kebijakan transisi** — scheduler (success-rate → naik stage) milik P3 training loop / P5 experiments.

**Belum (by design):** scheduler transisi (P3/P5) + sim-verify stage-2 spawn & stage-1 pre-grasp via `scripts/tune_arm.py`.

### Konsep
Task full kepanjangan untuk belajar dari nol → reward sparse → policy stuck. Pecah jadi tahap, isolasi tiap skill, gampang→susah.

### 4 stage (spec §7)

| Stage | Isi | Skill diisolasi | Status code |
|---|---|---|---|
| 1 Nav-only | box pre-grasped (`holding=1` saat spawn) | carry + place | ❌ `_sample_targets` selalu `holding=False` |
| 2 Grasp-only | spawn dekat box | approach + grasp | ❌ spawn selalu `x:-8..8, y:11..14` (receiving-north) |
| 3 Full chain | spawn jauh, nav→grasp→carry→place | gabung semua | ✅ default sekarang |
| 4 Anneal goal | `goal` xyz → 0, andalkan `goal_id`+pixels | deliver tanpa koordinat | ❌ `anneal_goal()` tidak dipanggil |

### Yang harus dibangun
1. **State** — `self.stage ∈ {1,2,3,4}` + `self.goal_alpha ∈ [0,1]` di `WarehouseRLEnv.__init__`
2. **Per-stage reset** di `_reset_idx` / `_sample_targets`:
   - stage 1 → `holding=True` + teleport box ke EE saat reset
   - stage 2 → override spawn pose dekat target box (bukan receiving-north)
   - stage 3 → perilaku sekarang
3. **Anneal** di `goal_position()` (`warehouse_env.py:99`) — kali `env.goal_alpha`
4. **API** — `env.set_stage(n)` + `env.set_goal_alpha(a)` supaya P3/P5 bisa atur dari training loop
5. **Scheduler** (siapa atur transisi: P3 training loop / P5 experiments) — mis. success-rate > threshold → naik stage. P4 sediakan mekanisme, bukan kebijakan.
6. **Test** — `tests/test_curriculum.py` tambah cek stage transitions + anneal

---

## ⚠️ Belum divalidasi: EE control tuning (spec §9)

Konstanta masih nilai awal, belum dikalibrasi di sim:

| Konstanta | Nilai | File | Risiko kalau salah |
|---|---|---|---|
| `EE_STEP_M` | 0.05 | `action_pickup.py` | kekecilan = arm lambat reach; kegedean = overshoot box |
| `GRIP_RADIUS_M` | 0.10 | `grasp.py` | kekecilan = susah grasp box besar; kegedean = false grasp |

**Tugas:** sesi tuning di sim — drive arm ke box tiap kategori (21/32/52 cm), catat nilai yang reliabel grasp tanpa false-positive.

---

## ⚠️ Perlu konfirmasi

### Arm reach end-to-end
Sesi lalu banyak fix arm (sag-gravity, relative-IK lokal — lihat `bugs_errors/2026-06-16_arm-sag-gravity-relative-ik.md`). **Belum diverifikasi** arm bisa reach + grasp box lengkap di sim. Tanpa ini, curriculum & tuning tidak bisa ditest.

### Carry model — DIPUTUSKAN: physics grasp (2026-06-20)
Tim pilih **physics grasp**, bukan kinematik teleport. Implementasi: `env/attach.py` weld box ke `panda_hand` pakai `UsdPhysics.FixedJoint` (pola sama dgn `_weld_robot_world_links`) saat grasp; lepas joint saat drop. Box dibawa di bawah physics (berat + collision), bukan disnap tiap step.
- `CARRY_MODE = "physics"` di `warehouse_env.py`; `"kinematic"` (teleport lama) tetap ada sebagai fallback.
- **Risiko + perlu sim-verify:** runtime add/remove joint di GPU PhysX pipeline kadang tak ke-pickup. Kalau box jatuh / error → balik ke `CARRY_MODE="kinematic"`. `scripts/tune_arm.py` print prim path `panda_hand` (harus non-None) untuk debug.

---

## Urutan kerja disarankan

1. **Konfirmasi arm reach** — blocker; tanpa ini tidak bisa test apa-apa
2. **EE tuning** — kalibrasi `EE_STEP_M`, `GRIP_RADIUS_M` setelah arm gerak
3. **Curriculum stage manager** — fitur terbesar yang hilang
4. **Keputusan carry kinematik** — tanya tim, dokumentasikan
