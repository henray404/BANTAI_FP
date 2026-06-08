# Visual Category-Aware World Model untuk Warehouse Pickup Robot
> Final Project Deep Learning (pure DL — NLP/PCD dropped 2026-06-08)
> Tim 5 Orang · 6 Minggu · RTX 5050 · Ubuntu / Windows 11

> ⚠️ **REDESIGN 2026-06-08:** CLIP (NLP) + YOLO (PCD) **dihapus**. Task: nav-only → **pick→carry→place**.
> Kategori box diberi langsung lewat `goal_id` one-hot (tanpa deteksi, tanpa teks). Lengan Franka AKTIF.
> CA-SLOPE + Visual HER **tetap** (CA-SLOPE baca kategori dari `goal_id`, bukan vision).
> Spec lengkap: **`docs/superpowers/specs/2026-06-08-pure-dl-pickup-redesign.md`**.

> ⚠️ **Dokumen ini punya 2 lapis:** bagian **"Status Implementasi"** di bawah = kondisi NYATA
> per 2026-06-08 (hasil audit kode). Sisanya = desain/visi (sebagian belum dibangun).
> Untuk jadwal realistis lihat **`docs/timeline_terbaru.md`**.

---

## Status Implementasi (per 2026-06-08 — audit kode, BUKAN rencana)

Ringkasan jujur kondisi workspace. Baca ini sebelum percaya bagian visi di bawahnya.

### ✅ Sudah ada (kode P1 — Environment, task LAMA = nav single-goal)
- Scene warehouse 20×30 m: 9 island / 18 rack, 54 shelf deck, 54 box rigid (gravity + massa), 3 zona, props, dinding, dome light. *(redesign: box turun ke ~18 graspable, satu shelf level)*
- **Robot = Ridgeback-Franka mobile manipulator** (base holonomik Clearpath + lengan Franka Panda 7-DOF + gripper) — **bukan** AMR diff-drive / Carter / Jetbot lagi.
- Base holonomik dipaksa diff-drive lewat mapping action `(2,)` → joint base (`_base_cmd`). *(redesign: action → `(6,)` tambah lengan+gripper)*
- Obs dict: `pixels, position, goal, goal_emb (zeros), heading` + Gymnasium wrapper. *(redesign: `goal_emb` → `goal_id(3)`, tambah `ee_pos, gripper, holding, box_pos`)*
- Reward nav: `delivery_success(+10)`, shaping jarak, time penalty, collision penalty. *(redesign: → staged pick-place, lihat spec §4)*
- Goal resampling per-env + randomisasi posisi box tiap reset. TiledCamera 64×64, contact sensor.
- Script: `run_env.py`, `drive_robot.py` (teleop, camera-strip), `smoke_test.py` (auto base test). Test: `test_env.py`, `test_layout_grid.py`, `test_obs.py`. `layout_grid.py` (math murni).

> ℹ️ Kode P1 di atas dibangun untuk task **nav single-goal lama**. Redesign 2026-06-08 (pickup) = migrasi env belum dikerjakan — lihat "Pickup migration" di `docs/environment.md` §12.

### ✅ Blocker TUNTAS (dulu 🔴, sekarang clear)
- **Camera SDP crash di RTX 5050 (Blackwell) RESOLVED** (2026-06-03) lewat downgrade driver NVIDIA 591.84 → 580.88 (DDU clean install) + 2 fix kode (contact-filter dihapus, `collision_penalty` shape). `test_env.py --num_envs 1` camera ON = **ALL PASS**. ⚠️ Pin driver di 580.88, jangan auto-update ke 591.x/595.x. Lihat `bugs_errors/2026-05-22_sdp-camera-crash-blackwell.md`.
- **Verifikasi first-run SELESAI:** frame prismatic = world-frame → `_base_cmd`/teleop proyeksi yaw; contact body = `base_link` (verified); box settle di shelf (z>0.5, fallen=0); VRAM muat. **Root-state frozen bug ditemukan + fixed** (2026-06-04): obs/reward baca `body_pos_w["base_link"]`, bukan root yang beku. `test_env.py` = **ALL PASS 16/16** + 3 regression check. Lihat `bugs_errors/2026-06-04_ridgeback-root-state-frozen.md`.
- **Sisa verify P1 (minor):** `run_env.py` windowed belum dites end-to-end; action smoothing/effort tuning sebelum training serius (base bisa keluar bounds <1 dtk @1.5 m/s).

### 🟡 Didesain tapi BELUM dibangun
- **Pickup task** (pick → carry → place pakai lengan). Spec approved 2026-06-08: `docs/superpowers/specs/2026-06-08-pure-dl-pickup-redesign.md` (revisi `2026-06-01-arm-pickup-design.md` tanpa YOLO/CLIP). Belum ada: arm IK (DifferentialIKController) wiring, obs `holding/box_pos`, reward staged. **Task aktual di kode sekarang = navigasi single-goal; pickup = migrasi belum jalan.**

### ❌ Belum ada sama sekali (P2–P5, seluruh stack ML)
- DreamerV3, CNN encoder, RSSM, replay buffer — **tidak ada di repo**.
- Category-Aware SLOPE, Visual HER — **tidak ada** (tetap in-scope, owner P5/P3 — lihat spec §4b).
- Baseline SAC/PPO, training pipeline, W&B, eksperimen — **tidak ada**.
- **Repo ini saat ini = 100% pekerjaan P1 (environment).**

> ❌ **CLIP + YOLO = DIHAPUS dari scope** (2026-06-08), bukan "belum dibangun". Kategori box dikirim langsung lewat `goal_id` one-hot.

### 📝 Status dokumentasi (per 2026-06-08, redesign pickup)
- `CLAUDE.md` → diupdate ke **pickup**: obs `goal_id`+manip keys, action `(6,)`, reward staged, roles baru, CA-SLOPE/HER.
- `docs/environment.md` → diupdate ke pickup: task, obs, action, reward, items rigid graspable, arm aktif, "Pickup migration" checklist §12.
- `docs/superpowers/specs/2026-06-08-pure-dl-pickup-redesign.md` → spec baru (source of truth).
- ⚠️ **`configs/env_config.yaml`** → masih state lama (`items.count: 54`, action 2-dim, nav reward). Belum diupdate ke pickup — perlu disinkronkan saat migrasi env.
- `docs/CHANGES.md` → changelog historis (entri Carter/box-physics/camera tetap; entri redesign pickup belum ditambah).

---

## Satu Kalimat

Robot gudang mobile-manipulator belajar mengambil box kategori yang diperintahkan dan mengantarnya ke zona warna yang benar — tanpa diprogram eksplisit, hanya dari pengalaman dan imajinasi.

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
[base_lin, base_ang, ee_dx, ee_dy, ee_dz, gripper]
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

**Warehouse Simulation** dibangun di Isaac Lab 5.1 dari NVIDIA. Layout: 9 island rak (3×3) di tengah scene dengan box graspable (~18, satu shelf level, dalam jangkauan lengan ~0.85m), tiga zona delivery berwarna di sisi shipping (selatan). Posisi box di-randomize setiap episode. Robot spawn random di area receiving (utara), yaw random.

Kategori box dikodekan **ukuran**, dan **diberi langsung ke robot lewat `goal_id` one-hot** (BUKAN dideteksi — YOLO/CLIP dihapus 2026-06-08):
- **21 cm → fragile** → zone_A (oranye `1.0,0.9,0.0`)
- **32 cm → regular** → zone_B (cyan `0.0,0.9,0.9`)
- **52 cm → heavy** → zone_C (ungu `0.7,0.0,0.9`)

Box diberi gradasi coklat (light/medium/dark). `goal_id` memilih sekaligus **box target** + **zona tujuan**. CA-SLOPE membaca kategori dari `goal_id` ini untuk reward landscape per-kategori.

> 🔧 **Robot:** **Ridgeback-Franka mobile manipulator** (base holonomik Clearpath + lengan Franka 7-DOF + gripper).
> Action space `(6,)` `[base_lin, base_ang, ee_dx, ee_dy, ee_dz, gripper]` — base holonomik dikekang jadi
> diff-drive; lengan **AKTIF** via DifferentialIKController (EE top-down) untuk **pickup task** (approved 2026-06-08).
> Kamera onboard 64×64 RGB. Locomotion + lengan + IK dari Isaac Lab, fokus proyek di learning algorithm.

---

## Task Hierarchy

Curriculum pickup (lihat spec §7) — robot mulai dari yang paling mudah:

**Curr 1 — Carry/Place** (box pre-grasped `holding=1` saat spawn): belajar bawa → lepas di zona. Isolasi delivery.
**Curr 2 — Grasp** (robot spawn di box): belajar approach → grasp. Isolasi manipulasi.
**Curr 3 — Full chain**: spawn receiving → navigate → grasp → carry → place. `goal` xyz masih diberi.
**Curr 4 — Anneal goal**: `goal` zona xyz anneal → zeros; robot andalkan `goal_id` + pixels. `box_pos` tetap diberi.

Reward staged auto-switch di flag `holding`: Phase A (approach+grasp, `+5` grasp) → Phase B (carry+place, `+10` delivery). Lihat spec §4.

> 🔧 **Status nyata (2026-06-08):** kode env masih task **nav single-goal lama** (Stage 1, terverifikasi
> end-to-end, `test_env.py` ALL PASS 16/16, camera resolved 2026-06-03). Curriculum pickup di atas =
> redesign baru di-spec 2026-06-08, **migrasi env belum dikerjakan**.

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
| P1 (Henry) | Environment & Integration (+ arm IK) | Isaac Lab scene, obs/action/reward, wiring DifferentialIKController, kamera |
| P2 | World Model core | CNN encoder, RSSM, decoder, world-model training |
| P3 | Policy + Visual HER | Actor-critic, replay buffer, training loop, HER relabel di buffer |
| P4 | Manipulation | Grasp detection, pick-place curriculum, EE control tuning (pair P1) |
| P5 | Experiments + CA-SLOPE | CA-SLOPE reward (RQ2 ablation), eval metrics, baseline SAC/PPO, W&B, paper |

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
    # navigation
    "pixels":   Tensor(batch, 3, 64, 64),   # kamera onboard
    "position": Tensor(batch, 3),            # posisi base robot xyz
    "heading":  Tensor(batch, 2),            # [cos(yaw), sin(yaw)]
    "goal":     Tensor(batch, 3),            # zona delivery xyz (anneal→zeros)
    "goal_id":  Tensor(batch, 3),            # one-hot [orange,cyan,purple] — pilih box + zona (ganti goal_emb 2026-06-08)
    # manipulation
    "ee_pos":   Tensor(batch, 3),            # end-effector xyz, base frame
    "gripper":  Tensor(batch, 1),            # bukaan finger 0..1
    "holding":  Tensor(batch, 1),            # 1.0 kalau box target ke-grasp
    "box_pos":  Tensor(batch, 3),            # box target xyz — UNANNEALED
}

action_space = Box(-1.0, 1.0, shape=(6,))
# [base_lin, base_ang, ee_dx, ee_dy, ee_dz, gripper]

buffer.add(obs, action, reward, next_obs, done)   # buffer BELUM ada di repo
buffer.her_relabel(trajectory)              # Visual HER (P3) — BELUM ada
buffer.sample(batch_size) → batch_dict
```

> 🔧 **Catatan:** obs `heading` sudah di kode lama. `goal_id` + 4 key manipulasi = redesign pickup
> (belum di kode). `buffer` + Visual HER belum diimplement.

---

## Stack Teknis

```
Simulator    : Isaac Lab 5.1 (NVIDIA)
Robot        : Ridgeback-Franka mobile manipulator (holonomik dipaksa diff-drive; Isaac 5.1 Nucleus USD)
World Model  : DreamerV3 (github.com/NM512/dreamerv3-torch)
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
