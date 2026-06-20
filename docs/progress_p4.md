# Progress P4 — Manipulation (pickup → carry → sort)

**Owner:** P4 (pairs w/ P1)
**Last updated:** 2026-06-20
**Status source:** dibangun + diverify sebagian di Isaac Sim (lihat catatan sim-verify)

Ringkasan apa yang **udah kelar** dan **apa yang kurang**, dipecah per bagian biar gampang dikerjain.

---

## ✅ Sudah kelar
- **Curriculum 4-stage** — env API (`set_stage`/`set_goal_alpha`, per-stage reset, goal anneal), unit-tested (`env/curriculum.py`, `env/warehouse_env.py`, `tests/test_curriculum.py`).
- **Physics grasp** — FixedJoint weld `panda_hand`↔box (`env/attach.py`); **kebukti jalan di sim** (box ke-grab, weld snap). `CARRY_MODE="physics"`, kinematic = fallback.
- **Magnetic pickup + arm beku** — `WarehouseGymEnv.step` nol-kan EE action → arm gak reach/knock; grasp by proximity (world-frame, `update_grasp`).
- **Grasp world-frame fix** — dulu bandingin EE base-frame vs box env-local (salah frame, gak nyala); sekarang konsisten world.
- **Demo script** — `scripts/demo_pickup.py` scripted base controller (drive→stop→grab→carry).
- **Tuning harness** — `scripts/tune_arm.py` (catatan: model lama, lihat A4).

---

## ❌ Apa yang kurang — per bagian

### A. Env manipulation (P4)
| # | Kurang | Aksi | Prioritas | Status |
|---|---|---|---|---|
| A1 | Stage-2 standoff kejauhan buat arm beku | turunin `_spawn_base_near_box` ~0.55/0.65 | tinggi | ✅ **DONE 2026-06-20** |
| A2 | `GRIP_RADIUS_M`/standoff belum dikalibrasi final | run sim, set angka reliable per kategori | sedang | ⬜ |
| A3 | Scheduler transisi stage (1→2→3→4) | **by design** punya P3/P5, bukan P4 | — | n/a |
| A4 | `tune_arm.py` usang (gerakin arm, model lama) | hapus / tandai deprecated | rendah | ⬜ |

### B. Demo / scripted nav
| # | Kurang | Aksi | Prioritas | Status |
|---|---|---|---|---|
| B1 | Demo nyangkut wall — nav lurus, gak hindar rak | waypoint manual muter rak, atau terima demo cuma buktiin grab | sedang | ⬜ |
| B2 | Obstacle avoidance asli | = hasil **policy ditraining** (baca pixels), bukan scripted | — | n/a |

### C. Policy P3 (`policy/actor_critic.py` — temannya)
| # | Kurang | Aksi | Prioritas | Status |
|---|---|---|---|---|
| C1 | Actor loss gak ada baseline → variance tinggi | `(returns - V).detach()` ganti raw returns | tinggi | ⬜ |
| C2 | Return normalization gak ada (DreamerV3 percentile) | tambah EMA 5–95% scaling | tinggi | ⬜ |
| C3 | REINFORCE, bukan dynamics-backprop | opsional, ganti ke reparam return | rendah | ⬜ |
| C4 | Entropy abai Jacobian tanh | minor, biarin | rendah | ⬜ |

### D. Sim-verify (cuma bisa di GPU sim run)
| # | Verify | Cara | Prioritas | Status |
|---|---|---|---|---|
| D1 | Physics weld bertahan full episode (box gak lepas pas carry) | `demo_pickup.py`, liat box nempel sampai zona | tinggi | ⬜ |
| D2 | Carry → drop zona warna bener (delivered) | demo + waypoint, atau training | sedang | ⬜ |
| D3 | `CARRY_MODE="kinematic"` fallback masih jalan | toggle, test | rendah | ⬜ |

---

## Urutan saran
1. ~~**A1** (standoff)~~ ✅
2. **C1 + C2** (policy baseline + return norm) — biggest impact ke training quality
3. **D1 / D2** (verify carry→sort di sim)
4. **B1** (waypoint demo) kalau mau presentasi visual
5. Bersihin **A4** (deprecate `tune_arm`)

---

## Konstanta penting (current)
| Konstanta | Nilai | File |
|---|---|---|
| `CARRY_MODE` | `"physics"` | `env/warehouse_env.py` |
| `GRIP_RADIUS_M` | `0.25` | `env/grasp.py` |
| `EE_STEP_M` | `0.05` (arm beku → tak terpakai) | `env/action_pickup.py` |
| Stage-2 standoff | `0.55` heavy / `0.65` kecil | `env/warehouse_env._spawn_base_near_box` |
