# Timeline Terbaru — Warehouse Robot World Model

> Disusun ulang dari **kondisi nyata workspace**, bukan dari rencana awal.
> Tanggal acuan: **2026-06-03** (Rabu).
> Sumber: audit kode `env/`, `scripts/`, `tests/`, `docs/`, git log, dan bug logs.
> Pendamping: lihat `docs/project_overview.md` (sudah diupdate) untuk status per komponen.

---

> ## ✅ UPDATE 2026-06-03 (sore) — BLOCKER CAMERA TUNTAS
> Camera SDP crash **RESOLVED**. Akar masalah = **driver NVIDIA** (sesuai riset). Setelah downgrade
> **591.84 → 580.88** + fix 2 bug kode (contact-filter, reward shape), `test_env.py --num_envs 1`
> camera ON = **ALL PASS (10/10)** — pertama kali env jalan end-to-end dengan camera.
> Critical-path `[BLOCKER]` di §2 **kelar**. Sekarang lanjut ke `[VERIFY]` first-run checks lalu `[ML]`.
> Detail: `bugs_errors/2026-05-22_sdp-camera-crash-blackwell.md` (Resolution), `docs/CHANGES.md`.
> ⚠️ **Pin driver di 580.88** — jangan auto-update ke 591.x/595.x (crash balik).

---

## 0. TL;DR (baca ini dulu)

- **Yang jalan:** environment P1 (scene, robot Ridgeback-Franka, reward nav, obs dict, Gym wrapper) — **kode + end-to-end run camera ON terverifikasi 2026-06-03 (ALL PASS)**.
- **~~Yang belum pernah terbukti jalan~~ → SUDAH JALAN:** env RL utuh end-to-end. Camera SDP crash **RESOLVED** (driver 580.88). `test_env.py` ALL PASS.
- **Yang belum ada sama sekali:** seluruh stack ML — DreamerV3, CLIP, YOLO, SLOPE, Visual HER, replay buffer, training pipeline, baseline. Repo ini **hanya berisi pekerjaan P1**.
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
[VERIFY] ✅ test_env.py --num_envs 1 (camera ON) = ALL PASS. run_env.py windowed: belum dites
      │
      ▼
[VERIFY] First-run checks: frame prismatic base, nama body contact sensor,
         box jatuh ke shelf, VRAM muat
      │
      ▼
[ML] DreamerV3 vanilla nyambung ke env  ←── P2 mulai serius di sini
      │
      ▼
[ML] Learning curve naik di task NAV-ONLY (Stage 1)
      │
      ▼
[ML] Tambah extension: CA-SLOPE / Visual HER + baseline SAC/PPO
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
- **[P1] Rapikan dokumen yang nyasar** (lihat §6) — `env_config.yaml` & `CLAUDE.md` masih nulis Carter.
- **[P2] Clone DreamerV3 (NM512/dreamerv3-torch) ke repo**, baca obs contract, siapkan adapter dict obs → input RSSM (handle key `heading`, `goal_emb` zeros). *Bisa mulai tanpa env render — pakai dummy obs generator dulu.*
- **Gate akhir Minggu 4:** `python tests/test_env.py --num_envs 1` → **ALL PASS** dengan camera nyala. Kalau tidak lolos, eskalasi Plan B (cloud) **sekarang**, jangan tunda.

### Minggu 5 — 11–17 Juni → **"World model pertama + baseline"**
- **[P2] DreamerV3 vanilla nyambung** ke `WarehouseGymEnv`, training loop jalan, learning curve mulai naik di **task NAV-ONLY** (Stage 1, reward yang sudah ada).
- **[P5] Baseline SAC/PPO** di env yang sama (model-free) untuk RQ1.
- **[P5] Setup W&B logging**, replay buffer (`buffer.add/sample`), seed control.
- **[P4] CLIP wiring** ke `goal_embedding()` (ganti placeholder zeros). Ringan, paralel.
- **[P3] YOLOv8**: dataset render dari env (butuh camera jalan), label per kategori box (size-coded). Geser ke Minggu 6 kalau camera baru beres telat.
- **Gate:** ada **minimal satu** learning curve (DreamerV3 atau SAC) yang naik di atas baseline acak.

### Minggu 6 — 18–24 Juni → **"Extension + eksperimen + paper"**
- **[P3-novelty] Category-Aware SLOPE** (reward shaping per kategori). Mulai dari SLOPE generic dulu, lalu kondisikan per kategori.
- **[P4-novelty] Visual HER** (relabel episode gagal berbasis kategori yang ter-approach).
- **[P5] Jalankan konfigurasi eksperimen** — realistis **3 dari 5** (bukan 5×3 seed), lihat §5.
- **[P5] Paper + demo video.**
- **Gate:** ada hasil yang bisa ditulis (curve perbandingan, success rate), walau ablation tidak lengkap.

### (Stretch / overflow) — setelah 24 Juni
- **Pickup task** (Ridgeback arm, `pickup_manager.py`, scripted IK, obs `carrying`) — sesuai `docs/superpowers/specs/2026-06-01-arm-pickup-design.md`. **Ini stretch goal, bukan jalur kritis untuk paper.**
- Seed tambahan, ablation penuh 5 konfigurasi.

---

## 4. Track Per-Orang (apa yang bisa dikerjakan paralel SEKARANG)

| Orang | Bisa mulai tanpa nunggu camera? | Aksi Minggu 4 |
|---|---|---|
| **P1 (Henry)** | — (dia yang pegang blocker) | Camera fix + verifikasi first-run + rapikan docs |
| **P2 DreamerV3** | ✅ pakai dummy obs | Clone repo, adapter obs dict (incl. `heading`), training loop skeleton |
| **P3 YOLO/SLOPE** | ⚠️ butuh frame untuk dataset | Siapkan pipeline label dari size-coded box; rancang quantile reward head |
| **P4 CLIP/HER** | ✅ sebagian | Implement `goal_embedding()` (CLIP frozen ViT-B/32, 512→64 proj); rancang relabel logic |
| **P5 Train/Paper** | ✅ | Setup W&B, replay buffer, struktur eksperimen, kerangka paper, mulai related-work |

**Prinsip:** P2/P4/P5 **jangan idle** nunggu camera. Pakai obs dummy + interface contract yang sudah fix untuk bangun kode mereka secara paralel. Yang benar-benar terblok camera cuma yang butuh frame asli (P3 dataset, validasi end-to-end).

---

## 5. Skenario Realistis (risk-adjusted)

Karena blocker camera punya ketidakpastian tinggi, ini 3 jalur:

### 🟢 Best case — camera beres < 3 hari
Ikuti §3 apa adanya. Akhir Juni: DreamerV3 + 1 novelty + baseline, paper dengan RQ1 + (RQ2 atau RQ3). 5 konfigurasi tetap tidak realistis full 3-seed; target 3 konfigurasi × 2 seed.

### 🟡 Likely case — camera beres dalam 1 minggu (butuh workaround/cloud)
Geser semua +1 minggu. **Descope:** pickup task → dibuang dari scope paper (jadi stretch). Fokus: nav-only + DreamerV3 + **satu** novelty (pilih CA-SLOPE, paling dekat ke kontribusi & paling murah dibanding HER). Baseline cukup SAC **atau** PPO, tidak dua-duanya.

### 🔴 Worst case — camera tidak bisa di RTX 5050 sama sekali
- **Plan B wajib:** pindah training ke GPU lain / cloud (Isaac Sim 5.1 headless, non-Blackwell). RTX 5050 lokal cuma untuk dev/teleop (camera-strip).
- Atau **descope drastis:** ganti `obs["pixels"]` jadi state-vector (posisi box + zona) sementara, world model jalan tanpa visual. **Tapi ini mengkhianati tesis "visual category-aware"** → hanya darurat untuk dapat *sesuatu* yang jalan demi nilai.

**Rekomendasi:** putuskan Best/Likely/Worst **di akhir Minggu 4** berdasarkan hasil camera fix. Jangan biarkan ketidakpastian ini menggantung sampai Minggu 6.

---

## 6. Yang Harus Segera Dirapikan (utang dokumentasi)

Dokumen-dokumen ini **bertentangan dengan kode** dan menyesatkan tim:

| File | Masalah | Aksi |
|---|---|---|
| `CLAUDE.md` | Tulis robot = Carter v1 diff-drive, misi = "nav-only Phase 1". Kode = Ridgeback-Franka, spec pickup sudah approved | Update robot + misi + obs contract |
| `configs/env_config.yaml` | `robot.type: carter_v1`, `items.count: 18 static_usd`, props forklift/palletasm | Ganti ke Ridgeback-Franka, 54 rigid boxes, props sesuai kode |
| `docs/CHANGES.md` | Mendokumentasikan switch ke **Carter v2.4** — kode sudah lewat itu ke Ridgeback-Franka | Tambah entri switch Carter→Ridgeback |
| `docs/project_overview.md` | Item color-coded (merah/hijau/biru), tidak sebut arm, task 3-stage seolah sudah ada | ✅ **sudah diupdate** (lihat section "Status Implementasi") |

> **Catatan working tree:** `env/warehouse_scene.py` (uncommitted) menambah kembali `visual_material` warna per-box (54 PreviewSurfaceCfg). Ini **bisa memicu lagi SDP crash** (komentar lama bilang 54 material node = trigger Blackwell crash). Verifikasi saat camera fix; kalau crash balik, revert ke versi tanpa per-box material.

---

## 7. Daftar Aksi Konkret (checklist, terurut prioritas)

1. [ ] **[P1]** Jalankan `smoke_test.py --headless` → catat verdict frame prismatic, nama body, VRAM.
2. [ ] **[P1]** Selesaikan / workaround camera SDP crash → `test_env.py` ALL PASS dengan camera nyala.
3. [ ] **[P1]** Verifikasi nama body contact sensor Ridgeback (ganti `base_link` kalau salah).
4. [ ] **[P1]** Verifikasi box settle di shelf deck.
5. [ ] **[P1]** Rapikan `env_config.yaml`, `CLAUDE.md`, `CHANGES.md` (§6).
6. [ ] **[P1]** Putuskan: working-tree per-box color di-keep atau revert (cek vs SDP crash).
7. [ ] **[P2]** Clone DreamerV3 ke repo, bikin adapter obs dict + training loop (pakai dummy obs dulu).
8. [ ] **[P5]** Setup W&B + replay buffer + seed control.
9. [ ] **[P4]** Implement `goal_embedding()` dengan CLIP frozen.
10. [ ] **[Tim]** Akhir Minggu 4: rapat keputusan skenario Best/Likely/Worst (§5) + descope final.

---

*File ini snapshot per 2026-06-03. Update tiap akhir minggu atau saat blocker camera berubah status.*
