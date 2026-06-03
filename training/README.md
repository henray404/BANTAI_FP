# training/ — Training Pipeline, Baselines & Experiments

> **Owner:** P5 (Coordinator / Training)
> **Deadline:** Minggu 5 (11–17 Juni) — W&B + baseline running; Minggu 6 — eksperimen + paper

---

## Tujuan

Setup seluruh training infrastructure: W&B logging, replay buffer, seed control,
baseline agents (SAC/PPO), dan experiment management untuk paper.

---

## Tugas Minggu 4 (4–10 Juni) — Infrastructure (✅ bisa mulai tanpa nunggu camera)

- [ ] Setup W&B project + logging wrapper
- [ ] Implement replay buffer (`buffer.add(obs, action, reward, next_obs, done)` → `buffer.sample(batch_size)`)
- [ ] Seed control (reproducibility: `torch.manual_seed`, `np.random.seed`, env seed)
- [ ] Struktur eksperimen: config per run, artifact logging
- [ ] Kerangka paper + mulai related work section

## Tugas Minggu 5 (11–17 Juni) — Baselines

- [ ] **Baseline SAC** di `WarehouseGymEnv` (model-free, untuk RQ1 comparison)
- [ ] **Baseline PPO** di `WarehouseGymEnv` (opsional, prioritas SAC dulu)
- [ ] Integrasi DreamerV3 (P2) ke training pipeline
- [ ] Learning curve logging ke W&B
- [ ] Gate: minimal satu learning curve naik di atas baseline acak

## Tugas Minggu 6 (18–24 Juni) — Experiments + Paper

- [ ] Jalankan konfigurasi eksperimen (realistis 3 dari 5, bukan 5×3 seed)
- [ ] Success rate, reward curves, ablation (sesuai scope)
- [ ] Paper: results section + demo video
- [ ] Gate: ada hasil yang bisa ditulis (curve perbandingan, success rate)

---

## Interface Environment

```python
from env.warehouse_env import WarehouseGymEnv, WarehouseEnvCfg

cfg = WarehouseEnvCfg()
env = WarehouseGymEnv(cfg=cfg)

obs, info = env.reset(seed=42)
# obs = dict dengan keys: pixels, position, goal, goal_emb, heading

action = env.action_space.sample()  # shape (2,), [-1, 1]
obs, reward, terminated, truncated, info = env.step(action)
# reward: Tensor(num_envs,)
# terminated/truncated: Tensor(num_envs,) bool
```

### Replay Buffer Interface (yang diharapkan P2)

```python
buffer.add(obs, action, reward, next_obs, done)
batch = buffer.sample(batch_size)
# batch.obs["pixels"], batch.obs["position"], etc.
```

---

## Konfigurasi Eksperimen (Target)

| ID | Config | Deskripsi | Owner |
|---|---|---|---|
| C1 | DreamerV3 vanilla | World model baseline | P2 |
| C2 | DreamerV3 + CA-SLOPE | Category-Aware SLOPE reward | P2 + P3 |
| C3 | DreamerV3 + Visual HER | Hindsight relabeling | P2 + P4 |
| C4 | SAC baseline | Model-free comparison | P5 |
| C5 | PPO baseline | Model-free comparison (opsional) | P5 |

**Realistis target:** 3 konfigurasi × 2 seeds (bukan 5×3).

---

## Struktur File yang Diharapkan

```
training/
├── README.md           # file ini
├── __init__.py
├── baselines/
│   ├── __init__.py
│   ├── sac.py          # SAC agent wrapper (stable-baselines3 atau custom)
│   └── ppo.py          # PPO agent wrapper
├── replay_buffer.py    # experience replay buffer
├── logger.py           # W&B logging wrapper
├── trainer.py          # main training loop (env interaction + agent update)
├── experiment.py       # experiment config + runner (multi-seed, multi-config)
├── seed.py             # seed control utilities
├── configs/            # per-experiment config files
│   ├── dreamerv3_vanilla.yaml
│   ├── dreamerv3_slope.yaml
│   ├── sac_baseline.yaml
│   └── ...
└── results/            # training outputs, checkpoints, curves
```

---

## Koordinasi

- **P1 (Henry):** env interface — kalau ada masalah `WarehouseGymEnv`, tanya Henry
- **P2 (DreamerV3):** training loop utama DreamerV3 bisa di `models/dreamerv3/` — replay buffer shared di sini
- **P3 (SLOPE):** SLOPE reward function masuk training loop sebagai auxiliary reward
- **P4 (Visual HER):** HER relabeling masuk replay buffer (`buffer.relabel()`)
