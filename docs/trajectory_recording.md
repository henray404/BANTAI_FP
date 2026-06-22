# Trajectory recording & replay — rekam run + demo replay terbaik (P5)

> Permintaan dosen: rekam **tiap gerakan robot** tiap menjalankan skenario, lalu pas demo **putar
> ulang run terbaik** dari CSV-nya. Ini sistemnya: rekam **lengkap** (semua sendi/kinematik + semua
> pose + event + metadata), replay **faithful** (set state terekam ke sim, bukan re-simulasi).

## Kenapa bukan sekadar log diagnostik

`scripts/drive_env.py --log` (punya teman) cuma nyimpen action + ee + euler base — itu **log
diagnostik teleop**, nggak cukup buat replay: nggak ada `joint_pos`, pose box, quaternion penuh,
maupun seed/skenario. Sistem ini ngerekam **semua** + metadata, dan ada **pemutarnya**.

## Yang direkam (tiap control step) — `runs/<name>.csv`

| Grup | Kolom |
|---|---|
| waktu | `step, t` |
| action (6) | `a_base_lin, a_base_ang, a_ee_dx, a_ee_dy, a_ee_dz, a_grip` |
| base_link (chassis bergerak) | `base_x/y/z`, `base_qw/qx/qy/qz`, `base_roll/pitch/yaw_deg` |
| end-effector | `ee_x/y/z`, `ee_qw..qz`, `ee_base_x/y/z` (base frame) |
| box target | `box_x/y/z`, `box_qw..qz` |
| grasp/goal | `gripper, holding, grasp_event, drop_event, goal_x/y/z` |
| reward/term | `reward, slope_reward, terminated, truncated, contact_force_n` |
| **kinematik** | **`q_<joint>` + `qd_<joint>` untuk SEMUA sendi** (base dummy x/y/z + Franka 7 + 2 finger) |

Kolom sendi dibuat **dinamis** dari `robot.joint_names` — nggak ada yang di-hardcode/dibuang.

## Metadata sekali per run — `runs/<name>.meta.json`

Biar run-nya bisa direkonstruksi & di-ranking: `seed`, `category/color/goal_id`, `target_box_name`,
`box_size_m`, `goal_zone_xyz`, `spawn_base_pose_w`, `env_origin`, `control_dt/hz`,
`max_episode_steps`, **`joint_names`** (urutan kolom untuk replay), `action_layout`, `slope_mode`,
dan `summary` (`success, return, steps, grasp_step, deliver_step`) untuk milih yang terbaik.

## Cara rekam (di mesin Isaac, bukan Mac)

```bash
conda activate isaaclab
python scripts/record_scenario.py --seed 0 --policy random --out runs/heavy_seed0
# berhenti otomatis saat delivered (--stop_on_success 1). --slope category buat rekam shaping juga.
```

Rekam banyak seed/skenario → tiap run hasilin `runs/<name>.csv` + `.meta.json`. Pilih **terbaik**
dari `summary` (mis. `success==1` & `steps` terkecil / `return` tertinggi).

**Teleop (nyetir manual) juga bisa rekam format replayable yang sama** — pakai `--record`:
```bash
python scripts/drive_env.py --record runs/teleop_best --seed 0
```
Rekam **satu episode** (sampai done), lalu berhenti; CSV+meta-nya bisa langsung di-replay. Flag `--log`
lama (CSV diagnostik) tetap ada dan terpisah — `--record` nggak ganggu itu. Tiga sumber run
(teleop / `record_scenario.py` random|checkpoint) semuanya hasilin format yang sama → satu
`replay_csv.py` buat semuanya.

## Cara replay buat demo (di mesin Isaac)

```bash
python scripts/replay_csv.py --run runs/heavy_seed0          # windowed, chase camera, pace asli
python scripts/replay_csv.py --run runs/heavy_seed0 --no_sleep
```

Replay nge-set `joint_pos`/`joint_vel` + pose box terekam ke sim tiap step (`write_joint_state_to_sim`
/ `write_root_state_to_sim`) lalu render — jadi **persis** run yang direkam, nggak tergantung
determinisme physics.

## ⚠️ Nyimpen run terbaik buat demo (penting)

`runs/` **ter-`.gitignore`** (baris 24), jadi rekaman nggak ke-commit. Supaya CSV terbaik kebawa
buat demo, pilih salah satu:
- tambah pengecualian: `!runs/demo_best/` di `.gitignore`, taruh best run di situ, atau
- copy ke folder ter-track, mis. `docs/demo/best_run.csv` + `.meta.json`.

Yang penting CSV + `.meta.json`-nya **sepasang** (replay butuh dua-duanya).

## Checkpoint state + auto-rewind (anti-stuck)

> Ini **checkpoint state simulasi**, BUKAN "policy checkpoint" (bobot tersimpan). Snapshot kondisi
> robot+box di milestone, lalu **rewind ke checkpoint terdekat** kalau robot ngaco.

**Snapshot diambil di** (`recording/checkpoint.py`):
- `approach_target` — pertama kali robot deket box (jarak < `approach_radius`, belum holding)
- `grasp` — saat box ke-grip (holding 0→1)
- `approach_delivery` — pertama kali box terbawa deket zona
- `progress` — **berbasis kemajuan, BUKAN tiap N langkah**: snapshot tiap kali robot **mendekat
  `--progress_delta` meter** (default **2.5 m**, pas buat gudang 20×30 m) ke target aktif (box saat
  approach, zona akhir saat carry). Kalau robot malah **menjauh, NGGAK ada snapshot** — jadi
  checkpoint terdekat selalu state progress-terbaik, nggak pernah mundur ke posisi lebih jauh.
- `periodic` — fallback interval tetap, **default OFF** (`--checkpoint_every 0`).

**Rewind ke checkpoint terdekat (paling baru) dipicu oleh:**
- `collision` — gaya kontak chassis > 5 N (nabrak rak/obstacle/box) → langsung rewind
- `drop` — box jatuh di tengah carry (`drop_event`) → rewind ke state saat box masih dipegang
- `out_of_bounds` — robot keluar ruangan (half-extent 9.5×14.5 m, samain `env.out_of_bounds`)
- `spinning` — base diam tapi yaw muter terus (muter di tempat yang sama)
- `idle` — base nyaris nggak gerak selama `--idle_seconds` (default 30 dtk) → robot bingung diam
- `no_progress` — robot **gerak tapi nggak makin dekat** ke target selama `no_progress_seconds`
  (default 45 dtk) → nutup kasus muter-muter di area luas yang lolos dari `idle`

**Eskalasi:** kalau rewind ke checkpoint yang sama gagal `escalate_after` kali (default 3), checkpoint
itu dibuang dan rewind mundur ke checkpoint sebelumnya — biar nggak loop selamanya di satu titik
jelek (`start` nggak pernah dibuang).

Cara restore: nulis balik `joint_pos`/`joint_vel` + pose box snapshot ke sim
(`recording/sim_state.py`) — kinematik dipulihkan persis, lalu lanjut dari situ. Tiap rewind dicatat
di CSV (kolom `checkpoint_event`, `restore_reason`) dan dihitung di `summary.n_rewinds`.

Aktif default saat rekam (`scripts/record_scenario.py --checkpoints 1`):
```bash
python scripts/record_scenario.py --seed 0 --policy random \
    --idle_seconds 30 --progress_delta 1.0 --out runs/heavy_seed0
```

Core logic-nya **diuji di Mac** lewat toy env (`tests/test_checkpoint.py`, 12 lulus: milestone,
progress (mendekat vs menjauh), no-periodic-default, idle, spinning, collision, drop, out-of-bounds,
no-progress, eskalasi, nearest, restore).

> ⚠️ Kalau policy-nya **deterministik**, restore ke state yang sama persis → robot ngulangin
> kegagalan yang sama → loop (mentok di eskalasi/`max_rewinds`). Buat rekam pakai `--policy random`
> aman (tiap restore beda). Pas policy asli (DreamerV3) masuk, andalkan stokastisitasnya atau tambah
> perturbasi kecil saat restore. Glue Isaac (`sim_state.py`) import-safe, perlu
smoke-test di mesin Isaac.

> Catatan: rekaman menangkap run **apa adanya termasuk rewind** (robot "loncat" balik di titik
> rewind, ditandai `restore_reason`). Kalau mau demo super mulus tanpa loncatan, rekam ulang sampai
> dapat run yang `n_rewinds == 0`, atau potong CSV di segmen bersih.

## Rekam per skenario + pilih run terbaik (metrik)

Tiap **run skenario** direkam jadi run sendiri, lalu di-skor **per skenario** (skenario = kategori
box: fragile/regular/heavy) untuk milih run demo terbaik.

**Metrik (per skenario):**
1. **success_rate** — proporsi episode yang sukses = box kategori benar terkirim ke zona warna benar
   lalu dilepas (env cuma men-trigger delivery di zona yang cocok, jadi `summary.success` sudah
   meng-encode ini).
2. **efisiensi sampel** — `steps_to_success` (langkah sampai delivered); makin sedikit makin efisien.
   (Versi kurva "langkah sampai capai ambang success-rate" itu metrik training di eval harness/W&B,
   bukan dari run demo — beda hal, jangan dicampur.)
3. **panjang episode** — `summary.steps` (langkah sampai selesai).

**Run terbaik per skenario** = run yang **sukses** dengan `steps_to_success` paling sedikit
(tie-break: return lebih tinggi). Ranking skenario: success_rate ↓, lalu episode lebih pendek.

**Rekam batch (env asli, Isaac):**
```bash
python scripts/record_scenario.py --policy random --seeds 0 1 2 3 4 --out runs/batch
# -> runs/batch/seed0.csv ... seed4.csv (+ .meta.json), dikelompokkan per kategori saat ranking
```

**Skor + pilih terbaik (jalan di Mac, nggak butuh Isaac):**
```bash
python scripts/rank_runs.py --dir runs/batch                    # tabel per skenario + winner
python scripts/rank_runs.py --dir runs/batch --copy_best docs/demo   # copy winner ke folder demo
```

Pipeline ini **terbukti end-to-end di Mac** lewat toy harness:
`python experiments/run_eval.py --record_dir /tmp/runs` (rekam tiap skenario×seed) →
`python scripts/rank_runs.py --dir /tmp/runs`. Logika skoring diuji di `tests/test_select_best.py`
(5 lulus). `--copy_best docs/demo` sekaligus jawab masalah `runs/` git-ignored (taruh winner di
folder ter-track).

## Peta file

| File | Isi |
|---|---|
| `recording/recorder.py` | `TrajectoryRecorder` (tulis CSV+meta) / `TrajectoryReader` (baca, rekonstruksi joint). Pure stdlib. |
| `recording/state_extractor.py` | Ambil state lengkap env asli → row datar + metadata. Import-safe. |
| `recording/replay.py` | Tulis state terekam balik ke sim (faithful playback). |
| `recording/checkpoint.py` | Milestone checkpoint + auto-rewind (idle/spin/collision/drop/oob/no-progress). Pure, testable. |
| `recording/sim_state.py` | Capture/restore state sim Isaac buat rewind. Import-safe. |
| `recording/select_best.py` | Skor run per skenario (success/efisiensi/panjang), pilih terbaik. Pure stdlib. |
| `scripts/record_scenario.py` | Rekam rollout skenario (+`--seeds` batch) + checkpoint rewind (Isaac). |
| `scripts/rank_runs.py` | Ranking run per skenario + pilih/copy run terbaik (Mac). |
| `scripts/replay_csv.py` | Putar ulang run (Isaac). |
| `tests/test_recording.py` | Round-trip recorder/reader (jalan di Mac). |

## Status & batas jujur

- Core recorder/reader **diuji di Mac** (`tests/test_recording.py`, 3 lulus) — plumbing + kolom
  sendi dinamis + rekonstruksi joint terbukti.
- `state_extractor`/`replay`/script **belum bisa dites di Mac** (butuh Isaac). Sudah dipastikan
  **import-safe** (nggak butuh torch/Isaac saat import), tapi WAJIB di-smoke-test di mesin Isaac:
  rekam 1 run → replay → cek robot+box gerak sesuai rekaman.
- `--policy checkpoint` masih **stub** (DreamerV3 actor butuh fitur RSSM dari P2 yang belum siap).
  Untuk sekarang rekam via `--policy random` atau teleop. Begitu policy siap, tinggal isi branch itu.
