# Visual Category-Aware World Model untuk Warehouse Robot
> Final Project Deep Learning  
> Tim 5 Orang · 6 Minggu · RTX 5050 · Ubuntu / Windows 11

> ⚠️ **Dokumen ini punya 2 lapis:** bagian **"Status Implementasi"** di bawah = kondisi NYATA
> per 2026-06-03 (hasil audit kode). Sisanya = desain/visi awal (sebagian belum dibangun,
> sebagian sudah berubah). Untuk jadwal realistis lihat **`docs/timeline_terbaru.md`**.

---

## Status Implementasi (per 2026-06-03 — audit kode, BUKAN rencana)

Ringkasan jujur kondisi workspace. Baca ini sebelum percaya bagian visi di bawahnya.

### ✅ Sudah ada (kode P1 — Environment)
- Scene warehouse 20×30 m: 9 island / 18 rack, 54 shelf deck, 54 box rigid (gravity + massa), 3 zona, props, dinding, dome light.
- **Robot = Ridgeback-Franka mobile manipulator** (base holonomik Clearpath + lengan Franka Panda 7-DOF + gripper) — **bukan** AMR diff-drive / Carter / Jetbot lagi.
- Base holonomik dipaksa diff-drive lewat mapping action `(2,)` → joint base (`_base_cmd`).
- Obs dict: `pixels, position, goal, goal_emb (zeros), heading` + Gymnasium wrapper.
- Reward nav: `delivery_success(+10)`, shaping jarak, time penalty, collision penalty. Termination: time_out, reached_goal, out_of_bounds.
- Goal resampling per-env + randomisasi posisi box tiap reset. TiledCamera 64×64, contact sensor.
- Script: `run_env.py`, `drive_robot.py` (teleop, camera-strip), `smoke_test.py` (auto base test). Test: `test_env.py`, `test_layout_grid.py`, `test_obs.py`. `layout_grid.py` (math murni).

### 🔴 Blocker / belum terbukti jalan
- **Camera SDP crash di RTX 5050 (Blackwell) masih OPEN.** Env RL butuh `pixels` → tidak bisa strip camera. `run_env.py` / `test_env.py` **belum pernah lolos end-to-end** di hardware ini. Hanya script camera-strip (teleop, smoke, explore) yang jalan. Lihat `bugs_errors/2026-05-22_sdp-camera-crash-blackwell.md`.
- **Verifikasi first-run belum dilakukan** (karena env belum bisa run penuh): frame prismatic base (world vs body), nama body contact sensor (`base_link` masih tebakan untuk Ridgeback), box settle di shelf, VRAM muat.

### 🟡 Didesain tapi BELUM dibangun
- **Pickup task** (pick → carry → deliver pakai lengan). Spec approved: `docs/superpowers/specs/2026-06-01-arm-pickup-design.md`. Belum ada: `pickup_manager.py`, STATION_SPECS, scripted IK, obs key `carrying`, reward pickup. **Task aktual sekarang = navigasi single-goal saja.**

### ❌ Belum ada sama sekali (P2–P5, seluruh stack ML)
- DreamerV3, CNN encoder, RSSM, replay buffer — **tidak ada di repo**.
- Category-Aware SLOPE, Visual HER — **tidak ada**.
- CLIP (`goal_emb` masih zeros), YOLOv8 — **tidak ada**.
- Baseline SAC/PPO, training pipeline, W&B, eksperimen — **tidak ada**.
- **Repo ini saat ini = 100% pekerjaan P1 (environment).**

### ⚠️ Utang dokumentasi (dokumen ≠ kode)
- `CLAUDE.md` → masih tulis Carter v1 + misi "nav-only Phase 1".
- `configs/env_config.yaml` → masih `carter_v1`, 18 box static, props forklift/palletasm.
- `docs/CHANGES.md` → dokumentasikan Carter **v2.4**, padahal kode sudah Ridgeback-Franka.
- Bagian "Environment dan Robot" + "Task Hierarchy" di dokumen ini sudah dikoreksi di bawah.

---

## Satu Kalimat

Robot gudang beroda belajar mengenali kategori item secara visual dan mengantar item ke zona yang benar — tanpa diprogram eksplisit, hanya dari pengalaman dan imajinasi.

---

## Masalah yang Diselesaikan

Robot gudang konvensional butuh jutaan percobaan nyata sebelum bisa belajar hal yang sederhana. Di warehouse nyata, ini artinya downtime produksi berhari-hari. Ditambah lagi, robot hanya tahu berhasil atau gagal di akhir episode — tidak ada sinyal di antaranya. Ini yang disebut **sparse reward problem**.

Ada tiga masalah spesifik yang saling berkaitan:

**Sample Inefficiency** — robot harus berinteraksi dengan environment jutaan kali sebelum policy-nya bagus. Di hardware nyata ini tidak praktis dan mahal.

**Sparse Reward** — robot hanya dapat reward +1 saat berhasil mengantar item ke tujuan yang benar. Selama ratusan ribu langkah sebelumnya, reward selalu 0 dan robot tidak tahu apakah gerakannya menuju arah yang benar atau tidak.

**Goal Ambiguity** — posisi item di-randomize setiap episode. Robot tidak bisa hardcode tujuannya. Ditambah lagi, episode yang gagal sebenarnya mengandung informasi berguna tentang apa yang berhasil dicapai robot — tapi metode standar membuang data ini begitu saja.

---

## Ide Utama

Daripada robot belajar dari trial & error langsung, robot membangun **model mental tentang dunianya** dari input kamera. Setelah model ini terbentuk, robot bisa berlatih ribuan kali di dalam imajinasinya sendiri tanpa perlu menyentuh environment nyata. Ini yang disebut **World Model**.

Untuk membuat World Model bekerja di kondisi sparse reward, dua teknik baru dikembangkan:

**Category-Aware SLOPE** mengubah cara robot merasakan seberapa dekat dia ke tujuan. Alih-alih satu landscape reward untuk semua goal, setiap kategori item punya landscape sendiri. Kalau goal-nya mengambil item merah (fragile), landscape mengarahkan robot ke arah item merah. Ini membuat gradient reward lebih informatif dan spesifik.

**Visual HER** memanfaatkan episode yang gagal. Kalau robot gagal mengantarkan item ke zona A tapi berhasil mendekati item biru (heavy), episode itu di-relabel sebagai sukses untuk task mengambil heavy item. Ini membuat setiap pengalaman robot — bahkan yang gagal — menjadi data belajar yang berguna.

---

## Bagaimana Sistem Bekerja

```
Episode dimulai
      │
      ▼
Robot melihat scene dari kamera onboard
Gambar 64×64 piksel, full color
      │
      ▼
CNN Encoder memproses gambar
menjadi representasi latent z yang kecil
"Aku lihat kotak merah di kiri, zona kuning di bawah"
      │
      ▼
RSSM (Recurrent State Space Model)
belajar memprediksi apa yang terjadi selanjutnya
"Kalau aku maju ke kiri, kotak merah akan lebih dekat"
      │
      ▼
Robot berlatih ribuan skenario di IMAJINASI
tanpa menyentuh environment nyata
      │
      ├── Category-Aware SLOPE
      │   membuat landscape reward per kategori item
      │
      └── Actor-Critic
          belajar aksi terbaik dari skenario imajinasi
      │
      ▼
Aksi dieksekusi di environment nyata
[linear_velocity, angular_velocity]
      │
      ▼
Kalau episode gagal → Visual HER relabel
"Kamu gagal zone A, tapi berhasil ambil fragile item
 → ini sukses untuk task pick fragile"
      │
      ▼
Data disimpan ke replay buffer
Loop kembali ke atas
```

---

## Komponen Teknis

### DreamerV3 — World Model Backbone

DreamerV3 dari Google DeepMind (Nature 2025) adalah algoritma yang memungkinkan robot membangun simulator di kepalanya sendiri dari input kamera. Arsitekturnya terdiri dari:

- **CNN Encoder** — memproses gambar 64×64 menjadi latent representation
- **RSSM** — Recurrent State Space Model dengan GRU dan stochastic component, memprediksi state berikutnya tanpa melihat environment nyata
- **Decoder** — merekonstruksi gambar dari latent untuk verifikasi kualitas world model
- **Actor-Critic** — belajar policy optimal sepenuhnya di dalam imajinasi world model

Dari 1.000 interaksi nyata, DreamerV3 bisa menghasilkan 100.000 langkah latihan di imajinasi. Efisiensi naik hingga 100x dibanding model-free RL.

### Category-Aware SLOPE — Kontribusi Utama

SLOPE (Li et al., Feb 2026) adalah teknik reward shaping berbasis potential function yang menggantikan reward biner dengan landscape gradual. Versi standar SLOPE menggunakan satu landscape untuk semua goal.

Kontribusi proyek ini: **Category-Aware SLOPE** mengkondisikan potential function pada kategori visual item yang jadi goal. Landscape reward untuk "ambil fragile item" berbeda dengan landscape untuk "ambil heavy item". Ini membuat sinyal reward lebih spesifik dan lebih informatif untuk multi-task setting.

### Visual HER — Kontribusi Utama

HER standar (Andrychowicz et al., NeurIPS 2017) merelabel episode gagal berdasarkan posisi yang dicapai robot. Kalau robot sampai ke titik X tapi goal-nya titik Y, HER relabel episode sebagai sukses untuk goal di titik X.

Kontribusi proyek ini: **Visual HER** merelabel berdasarkan kategori visual yang berhasil di-approach. Kalau robot mendekati kotak biru tapi gagal ke zona tujuannya, Visual HER relabel sebagai sukses untuk task yang melibatkan heavy item. Ini lebih semantically meaningful dibanding relabeling posisi biasa.

---

## Environment dan Robot

**Warehouse Simulation** dibangun di Isaac Lab 5.1 dari NVIDIA. Layout: 9 island rak (3×3) di tengah scene dengan box kardus di rak, tiga zona delivery berwarna di sisi shipping (selatan). Posisi box di-randomize setiap episode. Robot spawn random di area receiving (utara), yaw random.

> 🔧 **Koreksi vs kode (2026-06-03):** kategori item dikodekan **ukuran box**, bukan warna primer.
> Implementasi nyata di `env/warehouse_scene.py`:
> - **21 cm → fragile** → zone_A (oranye `1.0,0.9,0.0`)
> - **32 cm → regular** → zone_B (cyan `0.0,0.9,0.9`)
> - **52 cm → heavy** → zone_C (ungu `0.7,0.0,0.9`)
> Box diberi gradasi coklat (light/medium/dark), bukan merah/hijau/biru. Encoding ukuran ini
> yang dipakai YOLO/CLIP. (Skema warna merah/hijau/biru di paragraf di bawah = rencana lama.)

Item dikodekan dengan warna (rencana lama — TIDAK dipakai di kode):
- ~~Merah → fragile~~ → **diganti ukuran 21 cm**
- ~~Hijau → regular~~ → **diganti ukuran 32 cm**
- ~~Biru → heavy~~ → **diganti ukuran 52 cm**

Zona tujuan dikodekan dengan warna lantai:
- Oranye → zona A, tujuan fragile items (21 cm)
- Cyan → zona B, tujuan regular items (32 cm)
- Ungu → zona C, tujuan heavy items (52 cm)

> 🔧 **Koreksi robot (2026-06-03):** robot **bukan** AMR diff-drive murni lagi. Kode pakai
> **Ridgeback-Franka mobile manipulator** (base holonomik + lengan Franka 7-DOF + gripper).
> Action space tetap `(2,)` `[linear, angular]` — base holonomik sengaja dikekang jadi
> diff-drive supaya kontrak tidak berubah. Lengan ada untuk **pickup task** (masih spec, belum
> dibangun); saat ini lengan cuma diam di pose tucked. Kamera onboard 64×64 RGB. Locomotion +
> lengan dari Isaac Lab, fokus proyek di learning algorithm bukan kontrol robot.

---

## Task Hierarchy

Task dibagi tiga stage dengan curriculum — robot mulai dari yang paling mudah dan naik bertahap:

**Stage 1 — Navigate** ✅ *(SUDAH diimplement)*: robot bergerak dari spawn ke zona, menghindari rak dan dinding. Reward di kode: `+10` saat masuk radius zona (1.5 m) + shaping jarak.

**Stage 2 — Pick** 🟡 *(SPEC saja, belum dibangun)*: robot identifikasi box kategori benar dan pick pakai lengan Franka (scripted IK + kinematic attach). Detail: `docs/superpowers/specs/2026-06-01-arm-pickup-design.md`.

**Stage 3 — Deliver** 🟡 *(SPEC saja, belum dibangun)*: robot bawa box ke zona yang cocok, lepas. Reward pickup/deliver/mistake belum ada di kode.

> 🔧 **Status nyata:** hanya Stage 1 yang jalan (itupun belum terverifikasi end-to-end karena
> camera blocker). Stage 2–3 = desain mobile-manipulator yang baru di-spec 2026-06-01.

---

## Eksperimen

Lima konfigurasi dibandingkan dengan 3 random seed masing-masing:

| Konfigurasi | SLOPE | HER | Keterangan |
|---|---|---|---|
| SAC / PPO | ✗ | ✗ | Baseline model-free |
| DreamerV3 vanilla | ✗ | ✗ | World model tanpa extension |
| DreamerV3 + SLOPE generic | Generic | ✗ | Reward shaping standar |
| DreamerV3 + SLOPE + HER standar | Generic | Standar | Kedua teknik versi asli |
| **DreamerV3 + CA-SLOPE + Visual HER** | **Category-Aware** | **Visual** | **Proposed method** |

Metrik evaluasi: task success rate, sample efficiency (reward vs environment steps), episode length.

---

## Research Questions

**RQ1 — World Model**: Bagaimana DreamerV3 meningkatkan sample efficiency dibanding baseline model-free di sparse reward warehouse task?

**RQ2 — Category-Aware SLOPE**: Apakah mengkondisikan potential landscape pada kategori visual item meningkatkan learning speed dan success rate dibanding SLOPE generic?

**RQ3 — Visual HER**: Apakah relabeling berbasis visual achievement lebih efektif dibanding HER standar berbasis posisi di multi-task warehouse context?

---

## Pembagian Tim

| Person | Role | Tanggung Jawab |
|---|---|---|
| P1 | Simulation & Environment | Isaac Lab setup, warehouse scene, Gymnasium wrapper, kamera |
| P2 | DreamerV3 Base + Encoder | CNN encoder, RSSM, replay buffer, modifikasi arsitektur |
| P3 | Category-Aware SLOPE | Quantile reward head, potential landscape per kategori, QCE loss |
| P4 | Visual HER | Relabeling logic berbasis visual, goal representation, ablation |
| P5 | Training + Paper | Pipeline training, W&B monitoring, eksperimen, paper, demo video |

---

## Timeline

> ⚠️ **Tabel di bawah = rencana AWAL dan sudah TIDAK realistis** (per 2026-06-03 baru akhir
> Minggu 3, env belum bisa run end-to-end, stack ML belum ada). Jadwal yang dipakai sekarang:
> **`docs/timeline_terbaru.md`** — disusun ulang dari kondisi nyata + skenario risk-adjusted.

| Minggu | Fokus | Target (rencana awal) | Realita |
|---|---|---|---|
| 1 | Setup | Isaac Lab jalan, robot spawn, DreamerV3 clone | Setup ✅ · DreamerV3 belum di-clone ❌ |
| 2 | Environment | Scene selesai, Gym wrapper jalan | Kode ✅ · end-to-end ❌ (camera) |
| 3 | Baseline | SAC reward > 0, DreamerV3 vanilla mulai | ❌ tidak tercapai (waktu habis: ganti robot, box physics, debug camera) |
| 4 | World Model | DreamerV3 curve naik, stage 1 solve | belum mulai |
| 5 | SLOPE + HER | 5 konfigurasi × 3 seed selesai | tidak realistis |
| 6 | Paper + Demo | Paper final + demo video | tergantung blocker camera |

---

## Interface Contract (Tidak Boleh Diubah Tanpa Diskusi Tim)

```python
obs = {
    "pixels":   Tensor(batch, 3, 64, 64),   # kamera onboard
    "position": Tensor(batch, 3),            # posisi robot xyz
    "goal":     Tensor(batch, 3),            # posisi target xyz (anneal→zeros di curriculum)
    "goal_emb": Tensor(batch, 512),          # embedding goal (CLIP; masih zeros sampai P4)
    "heading":  Tensor(batch, 2),            # [cos(yaw), sin(yaw)] — SUDAH ada di kode (2026-06-01)
    # "carrying": Tensor(batch, 1),          # RENCANA pickup task (spec, belum ada di kode)
}

action_space = Box(-1.0, 1.0, shape=(2,))
# [linear_velocity, angular_velocity]

buffer.add(obs, action, reward, next_obs, done)   # buffer BELUM ada di repo
buffer.her_relabel(trajectory)              # Visual HER — BELUM ada
buffer.sample(batch_size) → batch_dict
```

> 🔧 **Catatan:** obs `heading` sudah ditambahkan ke kode (P2 harus handle di RSSM). `carrying`
> baru rencana pickup task. `buffer` + Visual HER belum diimplement sama sekali.

---

## Stack Teknis

```
Simulator    : Isaac Lab 5.x (NVIDIA)
Robot        : AMR differential drive (Isaac Lab built-in)
World Model  : DreamerV3 (github.com/NM512/dreamer-pytorch)
Reward       : Category-Aware SLOPE (custom extension)
Data Aug     : Visual HER (custom extension)
Training     : PyTorch, Weights & Biases
Hardware     : RTX 5050 8GB VRAM, 32GB RAM
OS           : Windows 11 / Ubuntu 22.04
Python       : 3.11, CUDA 12.8
```

---

## Novelty Statement

Belum ada paper yang mengkombinasikan tiga hal ini dalam satu sistem:

1. **Category-Aware SLOPE** — potential landscape yang dikondisikan pada kategori visual item, bukan hanya posisi
2. **Visual HER** — hindsight relabeling berbasis apa yang robot lihat dan dekati, bukan posisi geometris
3. **DreamerV3** sebagai backbone di warehouse multi-task navigation context

Masing-masing kontribusi bisa dievaluasi secara independen melalui ablation study, dan ketiganya saling memperkuat dalam satu pipeline end-to-end.

---

## Referensi Utama

- DreamerV3 — Hafner et al., Nature 2025 — https://arxiv.org/abs/2301.04104
- SLOPE — Li et al., Feb 2026 — https://arxiv.org/abs/2602.03201
- HER — Andrychowicz et al., NeurIPS 2017 — https://arxiv.org/abs/1707.01495
- DayDreamer — Wu et al., CoRL 2022 — https://arxiv.org/abs/2206.14176
- DreamerNav — Shanks et al., 2025 — https://pmc.ncbi.nlm.nih.gov/articles/PMC12510832
- SAC — Haarnoja et al., ICML 2018 — https://arxiv.org/abs/1801.01290
- RWARE — Christianos et al., 2020 — https://arxiv.org/abs/2006.07869
