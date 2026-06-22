# P5 Changes — CA-SLOPE, eval harness, trajectory recording & checkpoint rewind

> Changelog kerjaan P5 (reward shaping + eksperimen + tooling demo). Detail teknis ada di
> `docs/ca_slope.md` dan `docs/trajectory_recording.md`. CHANGES.md (P1) khusus perubahan env.

---

## 2026-06-22 — Checkpoint state + auto-rewind (anti-stuck)

Sistem **checkpoint state simulasi** (bukan policy/weights). Snapshot kondisi robot+box di milestone,
rewind ke checkpoint terdekat kalau robot ngaco. Detail: `docs/trajectory_recording.md` §Checkpoint.

**Snapshot diambil di:** `approach_target`, `grasp`, `approach_delivery`, dan **`progress`** —
berbasis kemajuan (tiap robot mendekat **`progress_delta` = 2.5 m** ke target aktif; gudang 20×30 m),
BUKAN tiap N langkah. Kalau robot menjauh → nggak ada snapshot, jadi checkpoint terdekat selalu
state progress-terbaik. `periodic` ada sebagai fallback, default OFF.

**Rewind dipicu oleh:** `collision` (>5 N), `drop` (box jatuh saat carry), `out_of_bounds` (keluar
ruangan), `spinning` (muter di tempat), `idle` (diam 30 dtk), `no_progress` (gerak tapi nggak makin
dekat 45 dtk). **Eskalasi:** kalau rewind ke checkpoint yang sama gagal `escalate_after` (3) kali,
checkpoint itu dibuang, mundur ke yang lebih lama (`start` nggak pernah dibuang).

**File:** `recording/checkpoint.py` (core, backend-agnostic), `recording/sim_state.py` (capture/restore
Isaac, import-safe), `experiments/toy_pickup_env.py` (+`capture`/`restore`), wiring di
`scripts/record_scenario.py` (`--checkpoints`, `--idle_seconds`, `--progress_delta`).
**Test:** `tests/test_checkpoint.py` — 12 lulus (milestone, progress, idle, spinning, collision, drop,
out-of-bounds, no-progress, eskalasi, nearest, restore) di Mac.

**Catatan:** logika terbukti di toy env (Mac); glue Isaac **belum di-smoke-test di Isaac**. Policy
deterministik + restore identik bisa loop — pakai `--policy random` atau perturbasi saat restore.

## 2026-06-22 — Full trajectory recording + faithful replay

Rekam **semua** gerakan robot per step ke CSV + metadata penuh, lalu **replay run terbaik** buat demo
(permintaan dosen). Detail: `docs/trajectory_recording.md`.

- Rekam per step: **semua joint** (`q_/qd_<joint>`, kinematik), pose base_link (xyz+quat+rpy), EE,
  pose box, gripper/holding/event, goal, action(6), reward, contact, terminasi.
- Metadata sekali (`<name>.meta.json`): seed, kategori/goal_id, box target+ukuran, spawn pose, zona,
  control_dt, `joint_names` (urutan replay), summary (success/return/steps/n_rewinds).
- Replay **faithful**: tulis joint+box state terekam ke sim tiap step (`write_joint_state_to_sim`),
  bukan re-simulasi → demo persis run yang direkam, lepas dari nondeterminisme physics.

**File:** `recording/recorder.py` (recorder/reader, pure stdlib), `recording/state_extractor.py`,
`recording/replay.py`, `scripts/record_scenario.py`, `scripts/replay_csv.py`.
**Test:** `tests/test_recording.py` — 3 lulus (round-trip, kolom joint dinamis, metadata) di Mac.

**Catatan:** core recorder/reader diuji di Mac; extractor/replay/script **belum di-smoke-test di
Isaac** (import-safe). `--policy checkpoint` masih stub (butuh world model P2). `runs/` git-ignored →
best run buat demo perlu folder ter-track (lihat §CSV terbuka di bawah).

## 2026-06-21 — CA-SLOPE reward shaping + headless eval harness

Metode P5 (RQ2): Category-Aware SLOPE, potential-based dense shaping per-kategori dari `goal_id`.
Detail + integrasi ke P2/P3: `docs/ca_slope.md`.

- `reward/ca_slope.py` — `CASlopeShaper` (`F=γΦ(s')−Φ(s)`, backend-agnostic numpy/torch), gain
  per-kategori 1.0/1.5/2.0, `category_aware=False` = generic (kontrol RQ2). `gamma=0.997` = `P3Config`.
- `reward/ca_slope_wrapper.py` — `CASlopeEnvWrapper`, integrasi anjuran: bungkus env, **nol perubahan
  P1/P2/P3**, baca `env.goal_pos` tak ter-anneal.
- `experiments/` — eval harness headless (toy env numpy + scripted policy placeholder), tulis
  `steps_<mode>.csv` (jejak per-step) + `summary_<mode>.csv` (performa per skenario), CLI `run_eval.py`.

**Test:** `tests/test_ca_slope.py` — 11 lulus (invariansi PBRS, category-aware, wrapper, mapping obs).

---

## 2026-06-22 — Teleop full recording (semua sumber replayable)

`scripts/drive_env.py` dapat flag **`--record <stem>`** + **`--seed`**: rekam **satu episode** teleop
ke format replayable lengkap yang sama (`<stem>.csv` + `.meta.json`), via `TrajectoryRecorder` +
`state_extractor.step_row`. Flag `--log` diagnostik lama **tetap ada & terpisah** (nggak diutak-atik).
Hasilnya: tiga sumber run — teleop, `record_scenario.py --policy random`, `--policy checkpoint` —
semua produksi CSV format sama → satu `scripts/replay_csv.py` buat semuanya. Compile OK; 26 test Mac
tetap lulus.

## 2026-06-22 — Per-scenario scoring + best-run selection (metrik dosen)

Rekam disambung ke run skenario + sistem skoring buat milih run demo terbaik per skenario.

- **`recording/select_best.py`** — skor run per skenario: (1) **success_rate** (box kategori benar →
  zona benar → dilepas), (2) **efisiensi sampel** (`steps_to_success`), (3) **panjang episode**
  (`steps`). Best per skenario = sukses + `steps_to_success` terkecil (tie-break return). Pure stdlib.
- **`scripts/rank_runs.py`** — CLI (Mac): tabel ranking per skenario + winner per skenario +
  `--copy_best <dir>` (copy winner ke folder ter-track → jawab masalah `runs/` git-ignored).
- **Sambungan ke run skenario:** `record_scenario.py --seeds 0 1 2 …` rekam batch (per kategori);
  `experiments/run_eval.py --record_dir <dir>` rekam tiap skenario×seed di toy harness.
- **Terbukti end-to-end di Mac** (toy): `run_eval --record_dir` (9 run) → `rank_runs` → winner per
  skenario + skenario terkuat. **Test:** `tests/test_select_best.py` — 5 lulus. Total P5 Mac: 31 lulus.

**Catatan:** "efisiensi sampel" di sini = steps-to-success per run demo. Versi kurva (env steps →
ambang success-rate) itu metrik training (eval harness/W&B), bukan dari run demo — beda hal.

## CSV — item terbuka (sisa)

1. ~~Teleop pakai recorder lengkap~~ — **DONE** (`drive_env.py --record`).
2. ~~`runs/` git-ignored → best run nggak ke-commit~~ — **DONE** (`rank_runs.py --copy_best docs/demo`,
   atau rekam langsung ke path ter-track). CSV + `.meta.json` harus sepasang.
3. ~~Index/ranking lintas run~~ — **DONE** (`select_best.py` + `rank_runs.py`).
4. Semua glue Isaac (`record_scenario.py`, `drive_env.py --record`, `sim_state.py`, `replay_csv.py`)
   **belum di-smoke-test di Isaac** — import-safe + compile OK, tapi wajib dijalanin sekali di mesin
   Isaac buat verifikasi end-to-end.
