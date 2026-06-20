# BANTAI_FP — Project Status (2026-06-20)

## Flowchart: Pipeline Training End-to-End

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                                │
│                                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │  Isaac    │───▶│  Warehouse   │───▶│   Episode    │───▶│  World    │ │
│  │  Sim 5.1  │    │  GymEnv (P4) │    │  Buffer (P5) │    │  Model    │ │
│  │  + Scene  │    │  obs v2 (9k) │    │  + HER       │    │  (P2)     │ │
│  │  ✅ DONE  │    │  act (6,)    │    │  ✅ DONE     │    │  ✅ DONE  │ │
│  └──────────┘    │  ✅ DONE     │    └──────┬───────┘    └─────┬─────┘ │
│                   └──────────────┘           │                  │       │
│                                              │    sample batch  │       │
│                                              ▼                  ▼       │
│                                    ┌──────────────────────────────┐     │
│                                    │      P3 Trainer              │     │
│                                    │  ┌────────┐  ┌───────────┐  │     │
│                                    │  │ Actor  │  │  Critic   │  │     │
│                                    │  │ ✅ DONE│  │  ✅ DONE  │  │     │
│                                    │  └────────┘  └───────────┘  │     │
│                                    │  train_loop    ✅ DONE      │     │
│                                    └──────────────┬──────────────┘     │
│                                                   │                     │
│                                                   ▼                     │
│                                    ┌──────────────────────────────┐     │
│                                    │     Logger (P5)  ✅ DONE     │     │
│                                    │  wandb / stdout fallback     │     │
│                                    └──────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
```

## Flowchart: World Model (P2) Internal

```
obs dict (9 keys)
  │
  ├─ pixels (B,3,64,64) ──▶ ConvEncoder (4-layer CNN) ──▶ 4096-dim  ─┐
  │                          ✅ DONE                                   │
  ├─ low_dim (B,19) ───────▶ MLPEncoder (2-layer MLP) ──▶ 256-dim   ─┤
  │   pos/head/goal/ee/...   ✅ DONE                                   │
  │                                                                     │
  │                                              concat ◀──────────────┘
  │                                                │
  │                                           embed (4352)
  │                                                │
  │                              action (B,6)      │
  │                                  │             │
  │                                  ▼             ▼
  │                           ┌─────────────────────────┐
  │                           │   RSSM  ✅ DONE         │
  │                           │  ┌─────────┐ ┌────────┐ │
  │                           │  │  GRU    │ │ Stoch  │ │
  │                           │  │ 512-dim │ │ 32×32  │ │
  │                           │  │ (deter) │ │ (1024) │ │
  │                           │  └─────────┘ └────────┘ │
  │                           │  feat = cat = 1536-dim  │
  │                           └────────────┬────────────┘
  │                                        │
  │                        ┌───────────────┼───────────────┐
  │                        ▼               ▼               ▼
  │                 ┌────────────┐  ┌────────────┐  ┌────────────┐
  │                 │ ConvDecoder│  │ RewardHead │  │  ContHead  │
  │                 │ (3,64,64)  │  │  scalar    │  │  sigmoid   │
  │                 │  ✅ DONE   │  │  ✅ DONE   │  │  ✅ DONE   │
  │                 └────────────┘  └────────────┘  └────────────┘
  │
  │  Loss = KL(post||prior) + recon + reward_mse + cont_bce
  │          ✅ dual KL        ✅ MSE    ✅ symlog    ✅ BCE
```

## Flowchart: Environment (P4) Internal

```
┌──────────────────── WarehouseGymEnv ────────────────────┐
│                                                          │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────┐ │
│  │   Scene     │   │  Obs Manager │   │ Action Mgr   │ │
│  │ 18 racks    │   │  9 obs terms │   │ base(3)      │ │
│  │ 18 boxes    │   │  ✅ DONE     │   │ arm_ik(3)    │ │
│  │ 3 zones     │   │              │   │ gripper(1)   │ │
│  │ Franka      │   │              │   │ ✅ DONE      │ │
│  │ ✅ DONE     │   │              │   │              │ │
│  └─────────────┘   └──────────────┘   └──────────────┘ │
│                                                          │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────┐ │
│  │ Reward Mgr  │   │ Termination  │   │ Curriculum   │ │
│  │ 7 terms:    │   │ 3 terms:     │   │              │ │
│  │  approach   │   │  time_out ✅ │   │ anneal_goal  │ │
│  │  grasp      │   │  success  ✅ │   │  ✅ DONE     │ │
│  │  carry      │   │  bounds   ✅ │   │              │ │
│  │  deliver    │   │  ✅ DONE     │   │ ⚠️  NOT      │ │
│  │  time_pen   │   │              │   │ WIRED INTO   │ │
│  │  collision  │   │              │   │ ENV CONFIG   │ │
│  │  drop       │   │              │   │              │ │
│  │  ✅ DONE    │   │              │   │              │ │
│  └─────────────┘   └──────────────┘   └──────────────┘ │
│                                                          │
│  ┌─────────────┐   ┌──────────────┐                     │
│  │ Grasp Model │   │  Visual HER  │                     │
│  │ proximity   │   │ relabel fail │                     │
│  │ + kinematic │   │ episodes     │                     │
│  │ carry       │   │ ✅ DONE      │                     │
│  │ ✅ DONE     │   │              │                     │
│  └─────────────┘   └──────────────┘                     │
└──────────────────────────────────────────────────────────┘
```

---

## Status per Komponen

### ✅ DONE (Production-Ready)

| Komponen | File | LOC | Keterangan |
|----------|------|-----|------------|
| **P4: Environment** | `env/warehouse_env.py` + 7 file | 1409 | Scene, obs v2, action 6D, reward staged, grasp, termination |
| **P2: World Model** | `models/dreamerv3/` (4 core file) | 814 | Encoder, RSSM, Decoder, RewardHead, ContHead, 16.3M params |
| **P3: Actor-Critic** | `policy/actor_critic.py` | 234 | Gaussian actor + tanh, slow-target critic, λ-return |
| **P3: Train Loop** | `policy/train_loop.py` | 399 | P3Trainer: collect→buffer→train WM+AC, eval, checkpoint |
| **P5: Logger** | `training/logger.py` | 73 | W&B + stdout fallback |
| **P5: Buffer** | `buffer/replay_buffer.py` | ~200 | EpisodeBuffer + Visual HER relabeling |
| **P5: Configs** | `policy/config.py` + yaml | ~160 | P3Config dataclass, env_config.yaml |
| **Tests** | `tests/` (14 file) | ~500 | ~70 unit test, semua passing |
| **Scripts** | `scripts/` (7 entry point) | ~300 | train_p3, train_dreamer, smoke_test, dll |

### ⚠️ ADA TAPI BELUM TERHUBUNG

| Komponen | File | Status | Yang Kurang |
|----------|------|--------|-------------|
| **Curriculum** | `env/curriculum.py` | Fungsi ada (`anneal_goal`, `goal_id_onehot`) | Belum di-wire ke `WarehouseEnvCfg` sebagai `CurriculumTermCfg`. Training jalan di full difficulty langsung |
| **Eval Loop** | `policy/train_loop.py` | `eval_episode()` ada (1 episode) | Batch evaluation + metrics aggregation belum ada |
| **Multi-env** | `env/warehouse_env.py` | Support `num_envs` | Fix di 1 env (VRAM 11GB RTX 2080 Ti limit) |
| **Sequence Batching** | `buffer/replay_buffer.py` | Single-transition only | RSSM `observe_sequence()` ada tapi buffer belum return trajectory |

### ❌ BELUM ADA

| Komponen | Keterangan | Prioritas |
|----------|------------|-----------|
| **CurriculumTermCfg di env config** | `anneal_goal()` ada tapi gak dipanggil otomatis tiap episode | Medium — training bisa jalan tanpa ini, tapi convergence lebih lambat |
| **Batch Eval + Metrics** | Evaluasi baru 1 episode, belum aggregate success rate, mean reward | Low — bisa ditambah kapan saja |
| **Perception (YOLO/CLIP)** | Dihapus by design — pure DL pakai goal_id one-hot | N/A — sengaja dihilangkan |
| **Real Robot Pipeline** | `drive_robot.py` skeleton ada, sensor pipeline belum | Low — fokus sim dulu |
| **Wandb Dashboard** | Logger siap, tapi wandb belum di-install/login | Low — stdout cukup dulu |

---

## Verified Working (2026-06-20)

```
✅ Isaac Sim 5.1 headless on RTX 2080 Ti (Turing sm_75)
✅ Robot loads, joints correct, fixed-base OK
✅ Camera rendering works (unlike Blackwell/RTX 5050)
✅ Mock WM training 3000 steps — EXIT=0
✅ Real WM training 200 steps — EXIT=0, checkpoint saved
✅ WM standalone test — all losses non-zero & decreasing:
     wm/loss:  3.45 → 1.27
     wm/kl:    3.42 → 1.40
     wm/recon: 0.37 → 0.19
     wm/reward: 0.32 → 0.23
     wm/cont:  0.71 → 0.005
✅ torch 2.7.0+cu128 pinned — no changes
✅ All unit tests passing
```

## Next Steps (Prioritized)

1. **Run longer training** (10k+ steps) — verify end-to-end convergence
2. **Wire curriculum** — `CurriculumTermCfg` in env config for progressive difficulty
3. **Install wandb** — `pip install wandb && wandb login` for proper metric tracking
4. **Batch eval** — aggregate success rate over N episodes
