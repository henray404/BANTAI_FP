# CA-SLOPE — Category-Aware SLOPE reward shaping (P5)

> Buat siapa pun yang jalanin training (P2 world model / P3 policy): baca ini dulu sebelum nyalain
> CA-SLOPE. Intinya — **CA-SLOPE itu cuma nambahin satu term ke reward**. Dia TIDAK ngubah obs,
> action, buffer, world model, atau actor-critic. Kamu cukup bungkus env-nya satu baris.

---

## 1. Apa ini & kenapa ada (RQ2)

**CA-SLOPE = dense reward shaping yang potential-based DAN per-kategori.** Tiap step kita tambahin

```
F(s, s') = gamma * Phi(s') - Phi(s)
```

ke reward env. `Phi(s)` ("potential") = seberapa dekat robot ke menyelesaikan task. Gain shaping-nya
**dibaca per-kategori dari `goal_id`** (one-hot fragile/regular/heavy — BUKAN dari vision; YOLO udah
dihapus 2026-06-08). Kategori berat dikasih landscape lebih curam (panduan lebih kuat) karena paling
susah dipelajari.

**RQ2:** apakah meng-kondisikan potential landscape pada kategori (CA-SLOPE) naikin learning speed /
success rate dibanding satu landscape generik untuk semua kategori? Itu sebabnya ada 3 mode:

| `mode` | Apa | Peran RQ2 |
|---|---|---|
| `category` | gain per-kategori (1.0 / 1.5 / 2.0) | metode usulan (CA-SLOPE) |
| `generic` | satu gain untuk semua (1.5) | kontrol |
| `none` | tanpa shaping (reward env apa adanya) | baseline |

## 2. Jaminan: shaping ini AMAN

Ini **potential-based reward shaping** (Ng, Harada & Russell 1999). Teoremanya: nambahin
`F = gamma*Phi(s') - Phi(s)` **tidak mengubah kebijakan optimal** untuk Phi apa pun. Jadi CA-SLOPE
mempercepat belajar TANPA mengubah task yang dipelajari — bukan reward hacking. (Lihat
`docs/research/referensi.md` #15 Ng1999, #16 Devlin2012.)

Konsekuensi praktis: kalau kamu liat `reward env` (base) vs `reward + shaping`, **base-nya identik**
di ketiga mode. Yang beda cuma sinyal dense-nya. Ini sengaja.

## 3. Yang dibaca & yang TIDAK disentuh

CA-SLOPE cuma butuh 5 hal, semua udah ada di env:

| Dibaca | Dari | Catatan |
|---|---|---|
| `goal_id` | `env.goal_id_buf` (N,3) | kategori one-hot → pilih gain |
| `ee_pos` | `env.ee_pos` (N,3) | Phase A: jarak ee→box |
| `box_pos` | `env.box_pos` (N,3) | tak ter-anneal |
| `goal_pos` | `env.goal_pos` (N,3) | **zona asli, TAK ter-anneal** — lihat ⚠️ di §5 |
| `holding` | `env.holding` (N,) | switch Phase A ↔ B |

**TIDAK disentuh sama sekali:** obs dict, action (6,), `EpisodeBuffer`, RSSM/encoder/decoder P2,
actor-critic P3, Visual HER. P2 & P3 jalan tanpa perubahan kode.

Potential-nya (lihat `reward/ca_slope.py`):

```
Phase A (belum holding): remaining = dist(ee, box) + phase_b_offset   # masih harus grasp + carry
Phase B (holding):       remaining = dist_xy(box, zona)               # tinggal carry + place
Phi = -gain[kategori] * remaining
```

`phase_b_offset` (~13 m, jarak baris-spawn → zona) bikin Phi nyaris kontinu saat `holding` flip 0→1,
jadi momen grasp nggak kena penalti shaping yang aneh.

## 4. Cara nyalain — bungkus env (CARA YANG DIANJURKAN)

Ini integrasi yang dipakai temen P2/P3. **Nggak ada perubahan di `train_loop.py` / world model /
buffer.** Cukup bungkus env sebelum dikasih ke `P3Trainer`:

```python
from env.warehouse_env import WarehouseGymEnv, WarehouseEnvCfg
from reward.ca_slope_wrapper import CASlopeEnvWrapper
from policy.train_loop import P3Trainer
from policy.config import P3Config

base_env = WarehouseGymEnv(WarehouseEnvCfg())
env = CASlopeEnvWrapper(base_env, mode="category")   # "generic" atau "none" untuk ablation RQ2

trainer = P3Trainer(env, world_model, cfg=P3Config(seed=0))
trainer.run(total_steps=200_000)
```

Reward yang udah di-shape ngalir lewat `env.step` → `buffer.add` → reward head world model →
imajinasi, persis kayak reward biasa. Wrapper juga naro `info["ca_slope_shaping"]` tiap step kalau
kamu mau log terpisah.

> ⚠️ **`gamma` HARUS sama** dengan diskon agen. `P3Config.gamma = 0.997` dan `CASlopeShaper` default
> `gamma=0.997` — udah cocok. Kalau kamu ganti salah satu, samain dua-duanya, kalau nggak jaminan
> invariansi PBRS-nya bocor.

### Alternatif: shaping di level batch (offline)

Kalau mau hitung shaping dari `buffer.sample()` (mis. buat eval/relabel, bukan di-inject ke training):

```python
from reward.ca_slope import CASlopeShaper
shaper = CASlopeShaper(category_aware=True)
batch = buffer.sample(256)
f = shaper.shaping_from_obs(batch.obs, batch.next_obs, done=batch.done)   # (B,)
shaped_reward = batch.reward + f
```

Ini mapping key obs v2 otomatis (`goal` → zona, `holding` (B,1) → (B,)). **Tapi baca §5.**

## 5. ⚠️ Jebakan yang WAJIB diketahui

**Anneal goal.** Di curriculum stage 4, obs key `goal` **anneal jadi nol**. Jadi kalau kamu hitung
Phase B dari `obs["goal"]` (jalur batch di atas), jaraknya jadi salah pas stage 4. **Solusi:** pakai
`CASlopeEnvWrapper` — dia baca `env.goal_pos` (zona asli, tak ter-anneal), bukan obs. Jalur batch
hanya valid sebelum anneal (stage 1–3). `box_pos` sendiri memang tak pernah di-anneal, aman.

**P2 obs adapter masih ketinggalan.** `models/dreamerv3/obs_adapter.py` saat ini cuma forward
`position|goal|goal_emb|heading` ke encoder (`goal_emb` itu CLIP lama yang harusnya udah dihapus,
belum termasuk `goal_id`/`box_pos`/`ee_pos`/`holding`). **Ini nggak ngeblok CA-SLOPE** — CA-SLOPE baca
state dari `env`, bukan dari fitur encoder P2. Tapi catat aja kalau lagi nyari kenapa world model
belum "lihat" kategori.

## 6. Status sekarang — KASARAN (jujur)

DreamerV3 (temen P2/P3) masih dibenerin dan Isaac Lab nggak jalan di Mac. Jadi yang bisa dijalanin
SEKARANG buat verifikasi adalah **eval harness headless** di `experiments/`:

- `experiments/toy_pickup_env.py` — pengganti kinematik numpy buat `WarehouseGymEnv`. Mereproduksi
  kontrak obs/action/reward, **bukan fisika**. Sengaja kasar.
- `experiments/scripted_policy.py` — policy greedy placeholder, **pengganti DreamerV3 actor** yang
  belum siap. Bukan policy hasil belajar — jangan dilaporin sebagai angka baseline.

Jalanin (cuma butuh numpy, nggak ada torch/Isaac):

```bash
python experiments/run_eval.py --ablation     # 3 mode back-to-back → CSV + tabel
```

Output di `training/results/eval/` (git-ignored): `steps_<mode>.csv` (rekam jejak tiap langkah robot)
+ `summary_<mode>.csv` (performa per skenario × seed). Detail kolom: `experiments/README.md`.

**Begitu stack asli siap, yang di-swap cuma dua, pipeline CSV/metrik tetap:**

| Sekarang (kasaran) | Nanti (asli) |
|---|---|
| `ToyPickupEnv` | `CASlopeEnvWrapper(WarehouseGymEnv(...))` |
| `ScriptedPickupPolicy` | DreamerV3 actor P3 (`trainer.actor.mean_action`) |

## 7. Peta file

| File | Isi |
|---|---|
| `reward/ca_slope.py` | Potential + shaping (backend-agnostic: numpy di Mac, torch di Isaac). |
| `reward/ca_slope_wrapper.py` | `CASlopeEnvWrapper` — integrasi anjuran (bungkus env). |
| `experiments/` | Eval harness headless + toy env + scripted policy + CLI. Lihat README di situ. |
| `tests/test_ca_slope.py` | Unit test (invariansi PBRS, category-aware, wrapper, mapping obs). |
| `docs/ca_slope.md` | Dokumen ini. |

## 8. Open items (belum kelar)

- [ ] Inject `CASlopeEnvWrapper` ke run DreamerV3 beneran begitu Isaac/policy siap, ukur RQ2.
- [ ] (Opsional) config yaml `training/configs/dreamerv3_slope.yaml` — sekarang parameter masih
      default Python di `CASlopeShaper`/`scenarios.py` (pyyaml belum keinstall di env Mac).
- [ ] Tuning `category_gains` & `phase_b_offset` ke skala reward env asli setelah ada data nyata.
- [ ] Koordinasi dgn P1: kalau mau CA-SLOPE jadi RewardTerm di `warehouse_reward.py` (bukan wrapper),
      samain titik injeksinya — tapi wrapper lebih disaranin karena nol perubahan ke P1/P2/P3.
