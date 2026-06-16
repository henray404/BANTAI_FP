# Timeline Terbaru — Warehouse Robot World Model

> Disusun ulang dari **kondisi nyata workspace**, bukan dari rencana awal.
> Tanggal acuan: **2026-06-08** (Senin).
> Sumber: audit kode `env/`, `scripts/`, `tests/`, `docs/`, git log, dan bug logs.
> Pendamping: lihat `docs/project/project_overview.md` (sudah diupdate) untuk status per komponen.

---

> ## 🔄 REDESIGN 2026-06-08 — SIMPLIFY: BUANG CLIP + YOLO, PICKUP JADI CORE
> **Scope berubah.** Proyek jadi **pure DL** (NLP/PCD dibuang). CLIP + YOLO **dihapus**.
> Kategori box diberi langsung lewat `goal_id` one-hot (tanpa deteksi/teks). **Lengan Franka AKTIF**
> (DifferentialIKController) — task naik dari nav-only ke **pick→carry→place**. Obs: `goal_emb(512)`
> → `goal_id(3)` + 4 key manipulasi (`ee_pos,gripper,holding,box_pos`). Action `(2,)` → `(6,)`.
> Episode 600 → 1000 step. **CA-SLOPE + Visual HER tetap** (CA-SLOPE owner P5 baca kategori dari
> `goal_id`; Visual HER owner P3 di replay buffer). Roles digeser: P3=Policy+HER, P4=Manipulation,
> P5=Experiments+CA-SLOPE. Spec: **`docs/superpowers/specs/2026-06-08-pure-dl-pickup-redesign.md`**.
> ⚠️ Item di bawah yang sebut CLIP/YOLO/nav-only/pickup-stretch = framing LAMA; pickup kini core,
> env P1 perlu migrasi (action 6-dim, arm IK, obs+reward pickup).

---

> ## 📌 UPDATE 2026-06-08 — STATUS RINGKAS
> **Env P1 = SELESAI & terverifikasi.** Camera blocker (06-03) + root-state frozen bug (06-04)
> sudah tuntas. `test_env.py --num_envs 1` (camera ON) = ALL PASS 16/16.
> **Utang dokumentasi §6 SUDAH LUNAS:** `CLAUDE.md`, `configs/env_config.yaml`, `docs/project/CHANGES.md`
> kini semua konsisten Ridgeback-Franka + 54 rigid box (bukan Carter lagi). Tabel §6 & checklist #5
> di bawah ditandai DONE.
> **Yang masih kosong = SELURUH stack ML (P2–P5).** Folder `models/dreamerv3`, `perception/detection`,
> `training` cuma berisi README + `__init__.py` — **0 baris kode fungsional**:
> tidak ada DreamerV3/RSSM, CA-SLOPE, Visual HER, replay buffer, W&B, baseline SAC/PPO, training
> loop, arm IK. **Jalur kritis sekarang = P2 mulai DreamerV3 + P1 migrasi env ke pickup.**
> (CLIP + YOLO **dihapus** dari scope 2026-06-08 — bukan lagi "yang kosong".)
> Sisa P1: migrasi pickup (action 6-dim, arm IK, obs+reward), `run_env.py` windowed belum dites,
> action smoothing/effort tuning, keputusan per-box color keep/revert.

---

> ## ✅ UPDATE 2026-06-03 (sore) — BLOCKER CAMERA TUNTAS
> Camera SDP crash **RESOLVED**. Akar masalah = **driver NVIDIA** (sesuai riset). Setelah downgrade
> **591.84 → 580.88** + fix 2 bug kode (contact-filter, reward shape), `test_env.py --num_envs 1`
> camera ON = **ALL PASS (10/10)** — pertama kali env jalan end-to-end dengan camera.
> Critical-path `[BLOCKER]` di §2 **kelar**. Sekarang lanjut ke `[VERIFY]` first-run checks lalu `[ML]`.
> Detail: `bugs_errors/2026-05-22_sdp-camera-crash-blackwell.md` (Resolution), `docs/project/CHANGES.md`.
> ⚠️ **Pin driver di 580.88** — jangan auto-update ke 591.x/595.x (crash balik).

---

> ## ✅ UPDATE 2026-06-04 — ROOT-STATE FROZEN BUG DITEMUKAN + DIPERBAIKI (P1)
> First-run `[VERIFY]` di §2 **kelar**. Ditemukan bug serius (sebelumnya tersembunyi): robot =
> fixed-root articulation, jadi `root_pos_w`/`root_quat_w` **beku di posisi spawn** sementara
> chassis (`base_link`) bergerak lewat dummy joints. Env membaca root di 3 tempat → `obs["position"]`,
> `obs["heading"]`, dan **semua** jarak reward/termination beku → navigasi tidak mungkin dipelajari.
> Plus: dummy prismatic = world-frame, jadi "maju" selalu geser ke world +x (abai heading).
> `test_env.py` lolos 10/10 dulu karena cuma cek **shape** obs, bukan apakah obs **berubah**.
> **Fix:** obs+reward baca `body_pos_w["base_link"]`; `_base_cmd`/teleop proyeksi yaw (body-frame).
> **Verified:** `test_env.py --num_envs 1` (camera ON) = **ALL PASS (16/16)** + 3 regression check baru.
> Detail + referensi: `bugs_errors/2026-06-04_ridgeback-root-state-frozen.md` (IsaacLab #1268, #2664).
> ⚠️ Catatan terpisah: base bisa keluar bounds <1 dtk @1.5 m/s + transien besar saat step velocity
> keras → perlu action smoothing / tuning effort sebelum training serius (di luar scope fix ini).
> **Next: `[ML]` DreamerV3 — env sekarang benar-benar belajar-able.**

---

## 0. TL;DR (baca ini dulu)

- **Yang jalan:** environment P1 (scene, robot Ridgeback-Franka, reward nav, obs dict, Gym wrapper) — **kode + end-to-end run camera ON terverifikasi 2026-06-03 (ALL PASS)**.
- **~~Yang belum pernah terbukti jalan~~ → SUDAH JALAN:** env RL utuh end-to-end. Camera SDP crash **RESOLVED** (driver 580.88). `test_env.py` ALL PASS.
- **Yang belum ada sama sekali:** seluruh stack ML — DreamerV3, CA-SLOPE, Visual HER, replay buffer, training pipeline, baseline, arm IK. Repo ini **hanya berisi pekerjaan P1**. (CLIP + YOLO dihapus dari scope 2026-06-08.)
- **Posisi waktu:** akhir Minggu 3 dari rencana 6 minggu. Rencana awal (Minggu 5 = 5 konfigurasi jalan) **sudah tidak realistis**, tapi blocker terberat baru saja hilang → momentum balik.
- **~~Risiko #1 eksistensial~~ (DICORET):** camera tidak render → proyek mati. **Sudah teratasi 2026-06-03.** Risiko #1 baru: stack ML (P2–P5) belum mulai sama sekali; P2 (DreamerV3) sekarang **unblocked** dan jadi jalur kritis.

---

## 1. Di Mana Kita Sekarang (akhir Minggu 3)

| Minggu | Rencana awal | Realita |
|---|---|---|
| 1 (14–20 Mei) | Setup, robot spawn, clone DreamerV3 | Setup Isaac Lab ✅. DreamerV3 belum di-clone ke repo ❌ |
| 2 (21–27 Mei) | Scene selesai, Gym wrapper jalan | Scene + wrapper **secara kode** ✅. End-to-end run ❌ (camera crash) |
| 3 (28 Mei–3 Jun) | Baseline dapat reward > 0, DreamerV3 vanilla mulai | **Tidak tercapai.** Waktu habis untuk: tukar robot ke Ridgeback-Franka, box physics (rigid + gravity), spec pickup, debugging camera |

**Kesimpulan:** kita **telat ~1–1.5 minggu** dari kurva ideal, dan blocker terberat (camera) belum tuntas. Robot juga baru saja diganti jadi mobile manipulator → menambah cakupan (arm) yang belum diimplement.

---

## 2. Critical Path (urutan WAJIB — tidak bisa dilompati)

```
[BLOCKER] ✅ DONE 2026-06-03 — Camera render di RTX 5050 (driver 580.88)
      │
      ▼
[VERIFY] ✅ test_env.py --num_envs 1 (camera ON) = ALL PASS 16/16. run_env.py windowed: belum dites
      │
      ▼
[VERIFY] ✅ First-run checks DONE: frame prismatic (world→yaw-proj fixed), contact body=base_link,
         box settle di shelf (z>0.5, fallen=0), VRAM muat. Root-state frozen bug fixed 06-04.
      │
      ▼
[MIGRASI] ⬅️ KITA DI SINI (P1). Env nav → pickup: action (6,), arm IK (DifferentialIKController),
         obs +manip keys, reward staged. Paralel: P2 DreamerV3 pakai dummy obs.
      │
      ▼
[ML] DreamerV3 vanilla nyambung ke env pickup  ←── P2 mulai serius (belum mulai)
      │
      ▼
[ML] Learning curve naik — curriculum: carry/place → grasp → full chain (lihat spec §7)
      │
      ▼
[ML] Tambah extension: CA-SLOPE (P5) / Visual HER (P3) + baseline SAC/PPO
      │
      ▼
[PAPER] Eksperimen multi-seed, ablation, paper + demo video
```

Semua kotak di bawah `[ML]` **menunggu** dua `[VERIFY]` selesai. Tidak ada gunanya P2–P5 ngebut kalau env belum bisa render frame.

---

## 3. Timeline Revisi (week-by-week, mulai sekarang)

> Asumsi deadline akhir proyek ~**akhir Juni 2026**. Kalau deadline berbeda, geser proporsional dan baca §5 (skenario).

### Minggu 4 — 4–10 Juni → **"Bikin env benar-benar jalan"**
Fokus **P1**, semua orang lain support / siapkan kode sambil nunggu.

- **[P1] Selesaikan camera SDP crash** (prioritas tunggal). Riset 2026-06-03 → **driver penyebab utama** (detail: `bugs_errors/2026-05-22_sdp-camera-crash-blackwell.md`). Opsi diurut:
  1. **DOWNGRADE driver NVIDIA 591.84 → 580.88** (validated Windows untuk Isaac Sim 5.1; known-good Blackwell 591.74). NVIDIA confirm branch 595.xx rusak di Blackwell; 591.84 juga di atas yang validated. **Coba ini DULU sebelum sentuh kode** (DDU clean install). Target: `test_env.py` PASS dgn camera ON.
  2. Kalau TiledCamera masih hang/crash: A/B test **standard `Camera`/`CameraCfg`** (IsaacLab #4951 — TiledCamera hang di Blackwell, Camera jalan utk num_envs=1).
  3. Turunkan tekanan VRAM (BAR1 8164/8192 hampir penuh; disable 54 box dekorasi + props saat bring-up camera nav-murni).
  4. **Plan B:** render/training di GPU lain / cloud (Isaac Sim headless di mesin non-Blackwell).
- **[P1] Jalankan `smoke_test.py`** → tentukan frame prismatic base (world vs body). Kalau world-frame, perbaiki `_base_cmd` (proyeksi yaw). *Catatan: ini bisa jalan sekarang karena camera di-strip — tidak nunggu blocker.*
- **[P1] Verifikasi nama body contact sensor** (`base_link` masih tebakan untuk Ridgeback) dari log articulation-init.
- **[P1] Verifikasi box jatuh ke shelf deck** (bukan ke lantai).
- **[P1] Rapikan dokumen yang nyasar** ✅ DONE (2026-06-08) — `env_config.yaml`, `CLAUDE.md`, `CHANGES.md` sudah sinkron Ridgeback-Franka (lihat §6).
- **[P2] Clone DreamerV3 (NM512/dreamerv3-torch) ke repo**, baca obs contract v2, siapkan adapter dict obs → input RSSM (handle `heading`, `goal_id(3)`, manip keys `ee_pos/gripper/holding/box_pos`). *Bisa mulai tanpa env render — pakai dummy obs generator dulu.*
- **Gate akhir Minggu 4:** `python tests/test_env.py --num_envs 1` → **ALL PASS** dengan camera nyala. Kalau tidak lolos, eskalasi Plan B (cloud) **sekarang**, jangan tunda.

### Minggu 5 — 11–17 Juni → **"World model pertama + baseline"**
- **[P2] DreamerV3 vanilla nyambung** ke `WarehouseGymEnv`, training loop jalan, learning curve mulai naik di **curriculum Curr 1** (carry/place, box pre-grasped).
- **[P3] Replay buffer** (`buffer.add/sample`) + actor-critic skeleton. Visual HER relabel nyusul.
- **[P4] Arm IK** (DifferentialIKController) + grasp-success detection — pair P1, copy `Isaac-Lift-Cube-Franka-v0`.
- **[P5] Baseline SAC/PPO** + setup W&B logging + seed control untuk RQ1.
- **Gate:** ada **minimal satu** learning curve (DreamerV3 atau SAC) yang naik di atas baseline acak.

### Minggu 6 — 18–24 Juni → **"Extension + eksperimen + paper"**
- **[P5-novelty] Category-Aware SLOPE** (reward shaping per kategori, baca kategori dari `goal_id`). Mulai dari SLOPE generic dulu, lalu kondisikan per kategori.
- **[P3-novelty] Visual HER** (relabel episode gagal berbasis kategori box yang ter-approach/grasp, di replay buffer).
- **[P5] Jalankan konfigurasi eksperimen** — realistis **3 dari 5** (bukan 5×3 seed), lihat §5.
- **[P5] Paper + demo video.**
- **Gate:** ada hasil yang bisa ditulis (curve perbandingan, success rate), walau ablation tidak lengkap.

### (Stretch / overflow) — setelah 24 Juni
- Curriculum penuh Curr 3–4 (full chain + anneal goal) kalau Curr 1–2 belum stabil tepat waktu.
- Seed tambahan, ablation penuh 5 konfigurasi.
- 6-DOF EE orientation / cuRobo motion planning (v1 = top-down + DifferentialIKController saja).

> ⚠️ **Pickup kini CORE** (bukan stretch) per redesign 2026-06-08. Tapi RISIKO TINGGI: pickup +
> arm IK + CA-SLOPE + Visual HER di atas env yang baru, dengan waktu mepet. Kalau telat, descope ke
> Curr 1 (carry/place, box pre-grasped) sebagai task minimum demi hasil yang bisa ditulis — lihat §5.

---

## 4. Track Per-Orang (apa yang bisa dikerjakan paralel SEKARANG)

| Orang | Bisa mulai tanpa nunggu camera? | Aksi Minggu 4 |
|---|---|---|
| **P1 (Henry)** | — (pegang migrasi env) | Migrasi pickup: action (6,), wiring arm IK, obs+reward staged; verify first-run |
| **P2 World Model** | ✅ pakai dummy obs | Clone DreamerV3, adapter obs dict (incl. manip keys), RSSM/encoder, training loop skeleton |
| **P3 Policy + Visual HER** | ✅ sebagian | Replay buffer (`buffer.add/sample`), actor-critic skeleton; rancang HER relabel di buffer |
| **P4 Manipulation** | ⚠️ butuh arm di env | Rancang grasp-success detection + pick-place curriculum; pair P1 utk DifferentialIKController |
| **P5 Experiments + CA-SLOPE** | ✅ | Setup W&B, baseline SAC/PPO, rancang CA-SLOPE (potential per kategori dari `goal_id`), kerangka paper |

**Prinsip:** P2/P3/P5 **jangan idle** nunggu migrasi env. Pakai obs dummy + interface contract (obs v2) yang sudah fix untuk bangun kode paralel. Yang terblok env-pickup: P4 (butuh arm IK live), validasi end-to-end.

---

## 5. Skenario Realistis (risk-adjusted)

Karena blocker camera punya ketidakpastian tinggi, ini 3 jalur:

### 🟢 Best case — camera beres < 3 hari
Ikuti §3 apa adanya. Akhir Juni: DreamerV3 + 1 novelty + baseline, paper dengan RQ1 + (RQ2 atau RQ3). 5 konfigurasi tetap tidak realistis full 3-seed; target 3 konfigurasi × 2 seed.

### 🟡 Likely case — camera beres dalam 1 minggu (butuh workaround/cloud)
Geser semua +1 minggu. **Descope:** pickup turun ke **Curr 1 saja** (carry/place, box pre-grasped — skip grasp/full-chain) supaya task minimum tetap pickup. Fokus: DreamerV3 + **satu** novelty (pilih CA-SLOPE, paling dekat ke kontribusi & paling murah dibanding HER). Baseline cukup SAC **atau** PPO, tidak dua-duanya.

### 🔴 Worst case — camera tidak bisa di RTX 5050 sama sekali
- **Plan B wajib:** pindah training ke GPU lain / cloud (Isaac Sim 5.1 headless, non-Blackwell). RTX 5050 lokal cuma untuk dev/teleop (camera-strip).
- Atau **descope drastis:** ganti `obs["pixels"]` jadi state-vector (posisi box + zona) sementara, world model jalan tanpa visual. **Tapi ini mengkhianati tesis "visual category-aware"** → hanya darurat untuk dapat *sesuatu* yang jalan demi nilai.

**Rekomendasi:** putuskan Best/Likely/Worst **di akhir Minggu 4** berdasarkan hasil camera fix. Jangan biarkan ketidakpastian ini menggantung sampai Minggu 6.

---

## 6. Yang Harus Segera Dirapikan (utang dokumentasi)

✅ **SEMUA LUNAS per 2026-06-08.** Dokumen-dokumen ini dulu bertentangan dengan kode; sekarang sudah sinkron:

| File | Masalah (dulu) | Status |
|---|---|---|
| `CLAUDE.md` | Tulis robot = Carter v1 diff-drive, misi = "nav-only Phase 1" | ✅ **DONE** — sudah Ridgeback-Franka + obs contract (incl. `heading`) + spec pickup |
| `configs/env_config.yaml` | `robot.type: carter_v1`, `items.count: 18 static_usd`, props forklift/palletasm | ✅ **DONE** — `type: ridgeback_franka`, `items.count: 54`, props sesuai kode |
| `docs/project/CHANGES.md` | Mendokumentasikan switch ke **Carter v2.4** saja | ✅ **DONE** — entri "Carter v2.4 → Ridgeback-Franka" + box-physics + camera-resolved + root-state-fix |
| `docs/project/project_overview.md` | Item color-coded, tidak sebut arm, task 3-stage seolah sudah ada, blocker camera OPEN | ✅ **DONE** — section "Status Implementasi" diperbarui 2026-06-08 |
| `docs/specs/environment.md` | §3 Robot masih Jetbot + wheel kinematics (WHEEL_BASE 0.118) | ✅ **DONE** (2026-06-08) — diganti Ridgeback-Franka `_base_cmd` |

> **Catatan working tree:** `env/warehouse_scene.py` (uncommitted) menambah kembali `visual_material` warna per-box (54 PreviewSurfaceCfg). Ini **bisa memicu lagi SDP crash** (komentar lama bilang 54 material node = trigger Blackwell crash). Verifikasi saat camera fix; kalau crash balik, revert ke versi tanpa per-box material.

---

## 7. Daftar Aksi Konkret (checklist, terurut prioritas)

1. [x] **[P1]** Frame prismatic **RESOLVED** = world-frame → `_base_cmd`/teleop proyeksi yaw (body-frame). Diverifikasi lewat regression `test_env.py` (forward-follows-heading cos=0.81). `smoke_test.py` sudah di-enhance (print root vs base_link) — jalankan standalone untuk verdict headless. Root-state frozen bug ditemukan+fixed (2026-06-04, lihat `bugs_errors/`).
2. [x] **[P1]** Camera SDP **DONE** (2026-06-03, driver 580.88) → `test_env.py --num_envs 1` **ALL PASS (16/16)** dengan camera + 3 regression check baru.
3. [x] **[P1]** Nama body contact sensor = `base_link` **VERIFIED** (smoke_test 2026-06-03; dipakai juga oleh obs/reward fix 2026-06-04).
4. [x] **[P1]** Box settle di shelf deck **VERIFIED** — `test_env.py` "Boxes remain on shelves (z > 0.5)" PASS (Fallen: 0).
5. [x] **[P1]** Rapikan `env_config.yaml`, `CLAUDE.md`, `CHANGES.md` (§6) — **DONE 2026-06-08** (+`environment.md` §3 robot).
6. [ ] **[P1]** Putuskan: working-tree per-box color di-keep atau revert (cek vs SDP crash).
7. [ ] **[P1]** Tes `run_env.py` windowed end-to-end (camera ON) — satu-satunya verify P1 yang belum.
8. [ ] **[P1]** Action smoothing / effort-velocity tuning sebelum training serius (base bisa keluar bounds <1 dtk @1.5 m/s).
9. [ ] **[P1]** Migrasi env nav → pickup: action (6,), arm IK (DifferentialIKController), obs `goal_id`+manip keys, reward staged. ← **JALUR KRITIS**
10. [ ] **[P2]** Clone DreamerV3 ke repo, bikin adapter obs dict v2 + training loop (pakai dummy obs dulu). ← **JALUR KRITIS**
11. [ ] **[P3]** Replay buffer (`buffer.add/sample`) + actor-critic; Visual HER relabel nyusul.
12. [ ] **[P4]** Grasp-success detection + pick-place curriculum (pair P1).
13. [ ] **[P5]** Setup W&B + baseline SAC/PPO + seed control; rancang CA-SLOPE.
14. [ ] **[Tim]** Rapat keputusan skenario Best/Likely/Worst (§5) + descope final.

---

*File ini snapshot per 2026-06-08 (redesign pickup: CLIP/YOLO dibuang, lengan aktif). Env P1 nav selesai; jalur kritis = P1 migrasi env→pickup + P2 DreamerV3. Update tiap akhir minggu.*
