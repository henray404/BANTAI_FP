# Progress P4 — Manipulation (pickup → carry → sort)

**Owner:** P4 (pairs w/ P1)
**Last updated:** 2026-06-21
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

## 🧊 Anti-freeze tuning (2026-06-21) — lawan robot "diam aja"

Keluhan: robot **freeze** (diam) buat ngehindarin collision penalty, bukannya navigate ngitarin rak.
6 perubahan, semua ke-wire:

| # | Perubahan | File | Nilai |
|---|---|---|---|
| 1 | **Curriculum mulai Stage 1** (nav-only, box pre-grasped) — belajar NAV + hindar rak dulu, grasp belakangan. Naik stage otomatis pas success-rate ≥ threshold | `policy/config.py` (`curriculum_start_stage=1`, window/threshold eksplisit), `policy/train_loop.py` (udah ke-wire) | start=1, window=20, thresh=0.6 |
| 2 | **Idle penalty** — diam ≥50 step (~5s) kena `-0.02`/step → diam LEBIH mahal dari gerak hati-hati | `env/warehouse_reward.py` (`idle_penalty`), `env/warehouse_env.py` (`RewardsCfg.idle`, `IDLE_PENALTY_STEPS=50`) | `-0.02`/step |
| 3 | **Collision diturunin** `-5 → -2` — biar robot berani eksplor (nabrak dikit pas belajar gpp) | `env/warehouse_env.py` (`collision` weight 5→2) | `-2`/step kontak |
| 4 | **Entropy dinaikin** `3e-4 → 1e-3` — robot coba muter obstacle, bukan diam | `policy/config.py` (`actor_entropy_scale`) | `1e-3` |
| 5 | **Training lama** 50k–200k step — obstacle avoidance muncul pelan via imagination rollout | `scripts/train_p3.py` (`--steps` default `200_000`) | default 200k |
| 6 | **Approach/carry pull DOUBLED** `-0.01 → -0.02` — dense draw ke goal ngalahin rasa takut collision | `env/warehouse_env.py` (`approach`/`carry` weight) | `-0.02`*dist |

**Trade-off** (diingetin user): collision kegedean = freeze, kekecilan = nabrak terus. `-2` titik tengah awal — naikin lagi kalau robot mulai nge-ram rak terus pas udah bisa navigate.

**Verify di sim** (belum di-run, butuh AppLauncher): `python scripts/train_p3.py --headless --steps 50000` → cek W&B `ep/reward` naik, `curriculum/stage` naik 1→2→3, robot gak diam di depan rak. Unit test pure-tensor: `14 passed` (`tests/test_curriculum.py`, `tests/test_reward_pickup.py`).

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
