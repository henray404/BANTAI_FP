# Referensi Paper, GitHub & Sumber Robot — Visual Goal-Conditioned World Model for Warehouse Pickup

**Project**: Visual Goal-Conditioned World Model for Warehouse Pickup (pure Deep Learning)
**Last updated**: 2026-06-17
**Status validasi**: di-research ulang oleh tim (3 researcher + supervisor) lalu link berisiko di-fetch ulang manual.
✅ = halaman abstract/repo di-fetch & dikonfirmasi sesi ini · ⚠️ = ada koreksi metadata · ★ = perkiraan star (tidak semua dihitung ulang).

> **CATATAN**: file ini dirombak total dari versi 2026-05-27 (framing lama "Text-Conditioned / CLIP + YOLO"). Sejak redesign 2026-06-08, CLIP (NLP) + YOLO (PCD) DIHAPUS → goal lewat `goal_id` one-hot. Referensi text-conditioned (LED-WM, LS-Imagine, RLVR-World, LM-Nav, Dreamwalker, VLA-MBPO) **di-drop** dari core; lihat §Dropped di bawah.

---

## A. Sumber Robot & Simulator (dari mana robotnya diambil)

Semua URL diverifikasi loads. Asset Ridgeback-Franka di-fetch ulang manual — path & joint names cocok verbatim dengan `CLAUDE.md`.

| # | Nama | URL resmi | What | Dipakai di projek |
|---|------|-----------|------|-------------------|
| 1 | **NVIDIA Isaac Lab** ✅ ★7.5k | https://github.com/isaac-sim/IsaacLab | Framework robot-learning GPU di atas Isaac Sim | Seluruh env (`warehouse_env.py`, scene, IK) dibangun di Isaac Lab 5.1 |
| 2 | **Isaac Lab Docs** ✅ | https://isaac-sim.github.io/IsaacLab/main/index.html | Dokumentasi resmi API | Referensi env cfg, controllers, assets |
| 3 | **NVIDIA Isaac Sim** ✅ | https://developer.nvidia.com/isaac/sim | Simulator robotika di atas Omniverse | Simulator/renderer tempat Isaac Lab jalan |
| 4 | **Ridgeback-Franka USD asset** ✅ | https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_assets/isaaclab_assets/robots/ridgeback_franka.py | Cfg `RIDGEBACK_FRANKA_PANDA_CFG`; USD: `{ISAAC_NUCLEUS_DIR}/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd` | **Sumber robot kita.** Joint `dummy_base_prismatic_x/y_joint`, `dummy_base_revolute_z_joint`, `panda_joint1..7`, `panda_finger_joint.*` cocok verbatim dgn CLAUDE.md. Asset di-stream dari Isaac Sim Nucleus. |
| 5 | **Clearpath Ridgeback** ✅ | https://clearpathrobotics.com/ridgeback-indoor-robot-platform/ | Base indoor holonomik (omni-drive Mecanum) | Asal hardware base holonomik yang disimulasikan |
| 6 | **Franka Robotics (Emika Panda)** ✅ | https://franka.de/ | Lengan 7-DOF force-sensitive (Panda → FR3) | Asal hardware lengan 7-DOF |
| 7 | **Isaac Lab reference envs** ✅ | https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html | `Isaac-Reach-Franka-v0`, `Isaac-Lift-Cube-Franka-v0` | Dua env acuan pola kontrol arm untuk task pick kita |
| 8 | **DifferentialIKController API** ✅ | https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.controllers.html | IK diferensial berbasis Jacobian | Driver arm: `(ee_dx,dy,dz)` → joint targets |

> Catatan: USD gabungan `ridgeback_franka.usd` tidak terdaftar di halaman public robot-assets Isaac Sim (di sana Clearpath + Franka terpisah). Provenans paling otoritatif = file cfg Isaac Lab (#4) — path Nucleus eksplisit + joint cocok.

---

## B. World Model Core (Tier 1)

### 1. DreamerV3 — Mastering Diverse Domains through World Models ✅
- Hafner, Pasukonis, Ba, Lillicrap — 2023 (arXiv), Nature 2025
- Paper: https://arxiv.org/abs/2301.04104
- Code (JAX official ✅ ★3.4k): https://github.com/danijar/dreamerv3
- Code (PyTorch ✅ ★860, **base P2**): https://github.com/NM512/dreamerv3-torch
- Relevansi: backbone seluruh projek. RSSM + actor-critic in imagination, symlog/two-hot, hyperparam tetap. **Wajib P2.**

### 2. Dream to Control — Dreamer/V1 ✅
- Hafner, Lillicrap, Ba, Norouzi — 2019 (ICLR 2020)
- Paper: https://arxiv.org/abs/1912.01603
- Relevansi: asal value-gradient lewat imagined latent rollouts; fondasi lineage RSSM.

### 3. DreamerV2 — Mastering Atari with Discrete World Models ✅
- Hafner, Lillicrap, Norouzi, Ba — 2020 (ICLR 2021)
- Paper: https://arxiv.org/abs/2010.02193
- Relevansi: latent kategorikal/diskret di RSSM yang diwarisi V3 — perlu untuk jelasin representasi world model kita.

### 4. DayDreamer — World Models for Physical Robot Learning ✅
- Wu, Escontrela, Hafner, Goldberg, Abbeel — 2022 (CoRL 2022)
- Paper: https://arxiv.org/abs/2206.14176
- Code (official): https://github.com/danijar/daydreamer
- Relevansi: Dreamer di lengan robot nyata (pick-place) + base beroda dari pixel + reward sparse — **precedent terdekat** untuk task mobile-manipulator pickup kita. Tier 1 (jembatan world-model ↔ manipulation).

### 5. TD-MPC2 — Scalable, Robust World Models for Continuous Control ✅
- Hansen, Su, Wang — 2023/2024 (ICLR 2024)
- Paper: https://arxiv.org/abs/2310.16828
- Code: https://github.com/nicklashansen/tdmpc2
- Relevansi: baseline model-based non-Dreamer untuk kontrol kontinu 6-DOF; pembanding kuat eksperimen P5.

### 6. IRIS — Transformers are Sample-Efficient World Models ✅
- Micheli, Alonso, Fleuret — 2022/2023 (ICLR 2023)
- Paper: https://arxiv.org/abs/2209.00588
- Code (✅ ★890): https://github.com/eloialonso/iris
- Relevansi: alternatif world-model berbasis transformer (vs RSSM) — sitasi tandingan saat memotivasi pilihan RSSM.

### 7. RoboDreamer — Learning Compositional World Models for Robot Imagination ✅ (Tier 3, related-work)
- Zhou, Du, Chen, Li, Yeung, Gan — 2024 (ICML 2024)
- Paper: https://arxiv.org/abs/2404.12377
- Relevansi: world model terkondisi-goal kompositional untuk manipulasi. Mendukung framing goal-conditioning, **tapi** berbasis video-generation/teks — kita pakai `goal_id` one-hot, jadi related-work saja.

---

## C. Manipulation & Mobile Manipulation (P4)

### 8. GAMMA — Graspability-Aware Mobile Manipulation Policy Learning ✅
- Zhang, Gireesh, Wang, Fang, Xu, Chen, Dai, He Wang — 2023/2024
- Paper: https://arxiv.org/abs/2309.15459 — "…based on Online Grasping Pose Fusion"
- Relevansi: mobile manipulation = navigasi + fusi grasp-pose; cocok dgn pipeline nav→grasp kita. Tier 2.

### 9. ReLMM — Fully Autonomous Real-World RL with Applications to Mobile Manipulation ✅
- Sun, Orbik, Devin, Yang, Gupta, Berseth, Levine — 2021 (CoRL 2021)
- Paper: https://arxiv.org/abs/2107.13545
- Relevansi: belajar navigasi + grasping bareng, policy termodularisasi — acuan kanonik RL nav+manip gabungan. Tier 2.

### 10. Continuously Improving Mobile Manipulation with Autonomous Real-World RL ✅
- Mendonca, Panov, Bucher, Wang, Pathak — 2024 (CoRL 2024)
- Paper: https://arxiv.org/abs/2409.20568
- Relevansi: framework mobile-manip RL otonom terbaru (Spot, ~80% sukses 4 task) — referensi state-of-the-art nav+manip. Tier 2.

---

## D. Metode RL — HER, Reward Shaping, Curriculum, Baseline

### Visual HER (P3)
**11. Hindsight Experience Replay (HER)** ✅ — Andrychowicz dkk. — 2017 (NeurIPS 2017)
- https://arxiv.org/abs/1707.01495 · Code: `HerReplayBuffer` di Stable-Baselines3
- Relevansi: **fondasi** Visual HER. Mekanik relabel episode-gagal-pakai-achieved-goal diambil dari sini. Tier 1.

**12. Addressing Sample Complexity in Visual Tasks Using HER and Hallucinatory GANs** ✅⚠️ — Sahni, Buckley, Abbeel, Kuzovkin — 2019 (NeurIPS 2019)
- https://arxiv.org/abs/1901.11529
- ⚠️ **Judul betul di atas**, BUKAN "Visual Hindsight Experience Replay" (itu cuma label arXiv v1). Sitasi pakai judul resmi.
- Relevansi: prior-work terdekat — extend HER ke domain visual di mana goal diinfer dari pixel. Tier 1.

**13. Visual RL with Imagined Goals (RIG)** ✅ — Nair, Pong, Dalal, Bahl, Lin, Levine — 2018 (NeurIPS 2018)
- https://arxiv.org/abs/1807.04742
- Relevansi: GCRL dari pixel mentah + relabel goal di latent — acuan kanonik relabel goal visual. Tier 1.

**14. Goal-Conditioned RL: Problems and Solutions (survey)** ✅ — Liu, Zhu, Zhang — 2022 (IJCAI-ECAI 2022)
- https://arxiv.org/abs/2201.08299
- Relevansi: survey framing representasi goal (termasuk one-hot/`goal_id`) + HER. Untuk problem framing. Tier 1.

### CA-SLOPE / Potential-Based Reward Shaping (P5)
**15. Policy Invariance Under Reward Transformations (PBRS)** — Ng, Harada, Russell — 1999 (ICML 1999)
- https://people.eecs.berkeley.edu/~russell/papers/ml99-shaping.ps (no arXiv; era pra-arXiv) · juga di Semantic Scholar
- Relevansi: **teorema fondasi.** `F(s,s') = γΦ(s') − Φ(s)` jaga policy optimal. Dense distance terkondisi-kategori kita = instance PBRS langsung. Sitasi wajib optimality-preservation. Tier 1.

**16. Dynamic Potential-Based Reward Shaping** — Devlin, Kudenko — 2012 (AAMAS 2012)
- https://eprints.whiterose.ac.uk/id/eprint/75121/
- Relevansi: extend bukti invariansi ke potensial yang berubah saat belajar — relevan bila CA-SLOPE terkondisi `goal_id` & bervariasi per fase. Tier 2.

### Curriculum (P4/P5)
**17. Curriculum Learning for RL Domains: A Framework and Survey** ✅ — Narvekar, Peng, Leonetti, Sinapov, Taylor, Stone — 2020 (JMLR 21)
- https://arxiv.org/abs/2003.04960
- Relevansi: survey kanonik curriculum RL. Untuk staging nav-only → grasp-only → full-chain → anneal-goal. Tier 1.

### Baseline (P5)
**18. Soft Actor-Critic (SAC)** ✅ — Haarnoja, Zhou, Abbeel, Levine — 2018 (ICML 2018) — https://arxiv.org/abs/1801.01290
**19. Proximal Policy Optimization (PPO)** ✅ — Schulman, Wolski, Dhariwal, Radford, Klimov — 2017 — https://arxiv.org/abs/1707.06347
- Baseline off-policy (SAC) + on-policy (PPO) untuk dibandingkan vs world-model.

---

## E. GitHub Repos

| Repo | ★ | Keterangan |
|------|----|-----------|
| https://github.com/NM512/dreamerv3-torch | ~860 | DreamerV3 PyTorch — **base P2** (README flag dirinya outdated, cek repo r2dreamer sebelum pin) |
| https://github.com/danijar/dreamerv3 | ~3.4k | DreamerV3 JAX official (author) |
| https://github.com/nicklashansen/tdmpc2 | — | TD-MPC2 official |
| https://github.com/eloialonso/iris | ~890 | IRIS transformer world model |
| https://github.com/danijar/daydreamer | — | DayDreamer robot |
| https://github.com/DLR-RM/stable-baselines3 | ~13.4k ✅ | SAC + PPO + `HerReplayBuffer` — repo baseline utama (P5/P3) |
| https://github.com/vwxyzjn/cleanrl | ~10k ✅ | SAC/PPO single-file readable — sekunder |
| https://github.com/isaac-sim/IsaacLab | ~7.5k ✅ | Framework env (P1) |

---

## F. Peta Baca per Person (pure-DL)

| Person | Prioritas |
|--------|-----------|
| **P1 (Env & Integration)** | Isaac Lab (A#1), DifferentialIKController (A#8), DayDreamer (#4) |
| **P2 (World Model)** | DreamerV3 (#1), Dreamer V1/V2 (#2,#3), IRIS (#6) |
| **P3 (Policy + Visual HER)** | HER (#11), 1901.11529 (#12), RIG (#13), GCRL survey (#14), SB3 |
| **P4 (Manipulation)** | GAMMA (#8), ReLMM (#9), Mendonca (#10), Lift-Cube env (A#7), DayDreamer (#4) |
| **P5 (Experiments/CA-SLOPE/baseline)** | Ng1999 (#15), Devlin2012 (#16), Curriculum survey (#17), SAC (#18), PPO (#19), TD-MPC2 (#5), SB3/CleanRL |

---

## G. Arsitektur Pipeline (pure-DL, tanpa CLIP/teks)

```
RGB obs (3x64x64) ──► CNN Encoder ─┐
                                   ├─► RSSM (h_t, z_t) ──► Actor-Critic ──► action (6,)
low-dim obs ───────────────────────┘                                       [base_lin, base_ang,
  position, heading, goal, goal_id(one-hot),                                ee_dx, ee_dy, ee_dz, gripper]
  ee_pos, gripper, holding, box_pos
```
- `goal_id` (3-dim one-hot) masuk RSSM langsung — **tanpa proyeksi CLIP** (512→64 dihapus 2026-06-08).
- DreamerV3 (#1) = core · DayDreamer (#4) = precedent robot · NM512 repo = implementasi P2.

---

## Dropped (dari versi 2026-05-27 — framing lama text-conditioned)

Di-drop karena CLIP/teks/YOLO dihapus, atau gagal validasi:
- **LED-WM** (arXiv 2511.22904) — language-aware Dreamer; framing teks dihapus + ID future-dated **tak terverifikasi**. ❌
- **VLA-MBPO** (arXiv 2603.20607) — ID future-dated **tak terverifikasi**. ❌
- **DreamerNav** (Frontiers/PMC) — tak di-fetch ulang; treat sebagai lead, konfirmasi sebelum sitasi. ⚠️
- **LS-Imagine, RLVR-World, LM-Nav, Dreamwalker, TransDreamer** — relevan ke framing teks/CLIP lama; bukan core pure-DL. Boleh related-work bila perlu.

> Catatan supervisor: ID arXiv ber-prefix 2511/2603/2605 = future-dated relatif tanggal sekarang; jangan sitasi tanpa fetch manual halaman abs.
