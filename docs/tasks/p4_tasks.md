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
| **Curriculum 4-stage** | ❌ **Belum** | helper ada, staging NOL — gap terbesar |
| **EE control tuning** | ⚠️ Belum divalidasi | konstanta masih nilai awal |
| **Arm reach end-to-end** | ⚠️ Perlu konfirmasi | banyak fix sesi lalu, belum di-verify lengkap |
| Carry kinematik vs physics | ⚠️ Perlu keputusan tim | sekarang teleport, bukan jepit fisik |

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

## ❌ Belum: Curriculum 4-stage (gap terbesar)

**Yang ada sekarang:** cuma helper di `env/curriculum.py`:
- `goal_id_onehot(cat_idx)` — dipakai di `_sample_targets` ✓
- `anneal_goal(goal_xyz, alpha)` — **TIDAK PERNAH dipanggil** ✗

**Yang hilang:** stage manager + wiring. Tidak ada `self.stage`, tidak ada scheduler; `_reset_idx` selalu jalan mode full-chain (`holding=False`, spawn receiving-north).

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

### Carry kinematik vs physics grasp
`_carry_held_boxes()` teleport box ke posisi EE tiap step (kinematik), bukan jepit fisik (friction gripper). Cukup untuk sinyal RL & sudah disengaja (box > bukaan gripper). **Keputusan tim:** apakah ini diterima untuk paper, atau perlu physics grasp? Spec tidak eksplisit minta physics — kemungkinan OK.

---

## Urutan kerja disarankan

1. **Konfirmasi arm reach** — blocker; tanpa ini tidak bisa test apa-apa
2. **EE tuning** — kalibrasi `EE_STEP_M`, `GRIP_RADIUS_M` setelah arm gerak
3. **Curriculum stage manager** — fitur terbesar yang hilang
4. **Keputusan carry kinematik** — tanya tim, dokumentasikan
