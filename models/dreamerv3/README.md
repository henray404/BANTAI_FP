# models/dreamerv3/ — World Model (DreamerV3)

> **Owner:** P2 (World Model)
> **Deadline:** Minggu 5 (11–17 Juni) — training loop jalan, learning curve mulai naik di NAV-ONLY

---

## Tujuan

Implementasi DreamerV3 world model (RSSM + actor-critic) yang nyambung langsung ke `WarehouseGymEnv`.
Referensi base code: [NM512/dreamerv3-torch](https://github.com/NM512/dreamerv3-torch).

---

## Tugas Minggu 4 (4–10 Juni) — Persiapan

- [ ] Clone / fork `NM512/dreamerv3-torch` ke folder ini
- [ ] Buat adapter obs dict → RSSM input (lihat contract di bawah)
- [ ] Handle semua obs keys: `pixels`, `position`, `goal`, `goal_emb`, `heading`
- [ ] `goal_emb` = zeros sampai P4 wiring CLIP — treat sebagai zero-filled placeholder
- [ ] Training loop skeleton (bisa pakai dummy obs generator dulu tanpa nunggu env)
- [ ] Pastikan bisa import `WarehouseGymEnv` dari `env/warehouse_env.py`

## Tugas Minggu 5 (11–17 Juni) — Training

- [ ] DreamerV3 vanilla nyambung ke `WarehouseGymEnv`
- [ ] Training loop jalan end-to-end
- [ ] Learning curve naik di task **NAV-ONLY** (Stage 1, reward yang sudah ada)
- [ ] Integrasi dengan W&B logging (koordinasi P5)

---

## Interface Contract (dari `env/warehouse_env.py`)

```python
# Observation space — DO NOT CHANGE tanpa diskusi tim
obs = {
    "pixels":   Tensor(batch, 3, 64, 64),   # RGB camera, float [0,1]
    "position": Tensor(batch, 3),            # robot xyz, env-local
    "goal":     Tensor(batch, 3),            # target zone xyz
    "goal_emb": Tensor(batch, 512),          # CLIP embedding (zeros sampai P4 wiring)
    "heading":  Tensor(batch, 2),            # [cos(yaw), sin(yaw)]
}

# Action space
action_space = Box(-1, 1, shape=(2,))       # [linear_vel, angular_vel]

# Reward
# success=+10, shaping=-0.01*dist, time=-0.005, collision=-5
```

### Catatan Penting

- **`heading` key** baru ditambah 2026-06-01 — wajib di-handle di RSSM input (gate with zero-weight kalau belum siap)
- **`goal_emb`** = zeros sampai P4 isi CLIP → jangan crash kalau isinya semua 0
- **`goal`** (xyz) akan di-anneal ke zeros di curriculum Phase 5 — jangan hardcode meaning-nya
- **`pixels`** = float [0,1], shape (batch, 3, 64, 64) — CHW format, bukan HWC

---

## Struktur File yang Diharapkan

```
models/dreamerv3/
├── README.md           # file ini
├── __init__.py
├── agent.py            # DreamerV3 agent (RSSM + actor-critic)
├── networks.py         # encoder, decoder, RSSM networks
├── replay_buffer.py    # experience replay (atau di training/)
├── obs_adapter.py      # adapter: env obs dict → RSSM input format
├── config.py           # DreamerV3 hyperparameters
└── utils.py            # helpers
```

---

## Cara Run Environment

```bash
conda activate isaaclab

# Test env bisa dipakai
python tests/test_env.py --num_envs 1

# Dummy obs (tanpa Isaac Sim) — buat development adapter
# Bikin sendiri: obs dict dgn shape sesuai contract, random values
```

---

## Koordinasi

- **P1 (Henry):** kalau ada masalah env / obs contract, tanya Henry
- **P4 (CLIP):** `goal_emb` akan diisi nanti — kamu handle zeros dulu
- **P5 (Training):** replay buffer bisa di sini atau di `training/` — koordinasi
