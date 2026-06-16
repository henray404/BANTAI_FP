# P3 Documentation — Policy (Actor-Critic + Training Loop) + Visual HER

**Author:** Jeremy (P3)  
**Last updated:** 2026-06-16  
**Project:** Visual Goal-Conditioned World Model for Warehouse Pickup (Isaac Lab)

---

## Table of Contents

1. [Overview](#overview)
2. [File Structure](#file-structure)
3. [Obs & Action Contract](#obs--action-contract)
4. [Data Flow](#data-flow)
5. [Module Reference](#module-reference)
   - [buffer/replay_buffer.py](#bufferreplay_bufferpy)
   - [buffer/visual_her.py](#buffervisual_herpy)
   - [policy/config.py](#policyconfigpy)
   - [policy/actor_critic.py](#policyactor_criticpy)
   - [policy/train_loop.py](#policytrain_looppy)
   - [scripts/train_p3.py](#scriptstrap_p3py)
6. [Integration Points](#integration-points)
   - [P1 (Environment)](#p1-environment)
   - [P2 (World Model)](#p2-world-model)
   - [P5 (Logger & Seed)](#p5-logger--seed)
7. [Running the Training Loop](#running-the-training-loop)
8. [Testing](#testing)
9. [Key Hyperparameters](#key-hyperparameters)
10. [Common Errors](#common-errors)

---

## Overview

P3 implements the **policy learning** layer on top of P2's DreamerV3 world model. The policy never trains directly on env observations — it trains on **imagined** RSSM feature trajectories produced by P2.

P3 owns:

| Component | Description |
|-----------|-------------|
| `buffer/replay_buffer.py` | Episode-tracking ring buffer, auto-applies HER on done |
| `buffer/visual_her.py` | Goal_id-based Visual HER relabeling (no CLIP, no language) |
| `policy/config.py` | `P3Config` dataclass — all hyperparameters |
| `policy/actor_critic.py` | `Actor`, `Critic`, `lambda_return` |
| `policy/train_loop.py` | `P3Trainer`, `WorldModelInterface` ABC |
| `scripts/train_p3.py` | Entry point — owns `AppLauncher` |
| `tests/test_p3_buffer.py` | Unit tests for buffer + HER integration |
| `tests/test_p3_actor_critic.py` | Unit tests for Actor, Critic, lambda_return |
| `tests/test_p3_visual_her.py` | Unit tests for Visual HER relabeling logic |

---

## File Structure

```
BANTAI_FP/
├── buffer/
│   ├── __init__.py
│   ├── replay_buffer.py       <- P3-owned EpisodeBuffer
│   └── visual_her.py          <- P3-owned Visual HER
├── policy/
│   ├── __init__.py
│   ├── actor_critic.py        <- Actor, Critic, lambda_return
│   ├── config.py              <- P3Config
│   └── train_loop.py          <- P3Trainer + WorldModelInterface
├── scripts/
│   └── train_p3.py            <- Entry point (AppLauncher owner)
└── tests/
    ├── test_p3_buffer.py
    ├── test_p3_actor_critic.py
    └── test_p3_visual_her.py
```

---

## Obs & Action Contract

P3 uses the **v2 contract** from `pembagian_tugas.md`. P1 must commit this before `scripts/train_p3.py` can run without `--mock_wm`.

### Observation (9 keys)

| Key | Shape | Description |
|-----|-------|-------------|
| `pixels` | `(3, 64, 64)` | RGB camera feed (TiledCameraCfg) |
| `position` | `(3,)` | Robot base xyz in world frame |
| `heading` | `(2,)` | `[cos theta, sin theta]` robot yaw |
| `goal` | `(3,)` | Target zone xyz |
| `goal_id` | `(3,)` | One-hot zone category (0=orange/fragile, 1=cyan/regular, 2=purple/heavy) |
| `ee_pos` | `(3,)` | End-effector xyz |
| `gripper` | `(1,)` | Gripper opening [0,1] |
| `holding` | `(1,)` | Whether box is held: >= 0.5 means grasped |
| `box_pos` | `(3,)` | Box xyz in world frame |

### Action (6,)

```
[base_lin, base_ang, ee_dx, ee_dy, ee_dz, gripper]
```

All values in `[-1, 1]`. Mapped by P1 to actual Isaac Lab actuator commands.

### Zone Mapping

```
zone_A: goal_id=[1,0,0]  xyz=(-6, -12, 0.01)  orange / fragile
zone_B: goal_id=[0,1,0]  xyz=( 0, -12, 0.01)  cyan   / regular
zone_C: goal_id=[0,0,1]  xyz=( 6, -12, 0.01)  purple / heavy
```

---

## Data Flow

```
Isaac Lab env (P1)
       |  obs_v2 (9 keys), action (6,)
       v
EpisodeBuffer  ---on done--->  VisualHER relabels trajectory
       |                       adds relabeled transitions back to buffer
       |  Batch (obs, action, reward, next_obs, done)
       v
WorldModel.train_batch()  (P2 -- DreamerV3)
       |
WorldModel.encode_obs()   -> start_feat (B, 1536)
       |
imagination loop (H steps):
  Actor -> action_t -> WorldModel.imagine_step() -> next_feat, pred_r, pred_c
       |
  feats  (H+1, B, 1536)
  rewards (H, B)
  conts   (H, B)
       |
  Critic.slow_value() -> V_slow (H+1, B)
  lambda_return(rewards, V_slow, conts) -> returns (H, B)
       |
  Critic.loss(feats[:-1], returns) -> MSE -> critic_opt.step()
  Actor.loss(feats, returns) -> policy gradient + entropy -> actor_opt.step()
       |
  Logger.log(metrics)  (P5 W&B)
```

---

## Module Reference

### buffer/replay_buffer.py

**`EpisodeBuffer(capacity, her_fn=None, seed=0)`**

Ring buffer with episode tracking. Automatically calls `her_fn(trajectory)` and stores relabeled transitions when `done=True` is received.

| Method | Signature | Description |
|--------|-----------|-------------|
| `add` | `(obs, action, reward, next_obs, done)` | Accepts single step or batched `(B,...)` inputs |
| `sample` | `(batch_size) -> Batch` | Random sample; raises `RuntimeError` if empty |
| `__len__` | -- | Number of valid transitions stored |

**`Batch` (dataclass)**

```python
@dataclass
class Batch:
    obs:      dict[str, np.ndarray]   # v2 keys, each (B, ...)
    action:   np.ndarray              # (B, 6)
    reward:   np.ndarray              # (B,)
    next_obs: dict[str, np.ndarray]   # same keys as obs
    done:     np.ndarray              # (B,) bool
```

**Usage example:**

```python
from buffer import EpisodeBuffer, make_visual_her_fn, ZONE_POSITIONS

her_fn = make_visual_her_fn(ZONE_POSITIONS, success_reward=10.0, her_ratio=0.5)
buf = EpisodeBuffer(capacity=100_000, her_fn=her_fn, seed=0)

# Inside env loop:
buf.add(obs, action, reward, next_obs, done)  # HER fires automatically on done=True
batch = buf.sample(16)
```

---

### buffer/visual_her.py

**`make_visual_her_fn(zone_positions, success_reward=10.0, her_ratio=0.5)`**

Returns a relabeling function `fn(trajectory) -> list[dict]`.

**Algorithm:**

1. Gate: `if random() > her_ratio: return []`
2. Find `grasp_idx`: first step where `next_obs["holding"] >= 0.5`
3. If no grasp found: return `[]`
4. Find nearest zone to robot `position` over all post-grasp steps (2D L2 distance)
5. For each step in `trajectory[grasp_idx:]`:
   - Set `obs["goal"] = zone_pos[nearest_zone_idx]`
   - Set `obs["goal_id"] = ZONE_GOAL_IDS[nearest_zone_idx]` (one-hot)
   - Same for `next_obs`
   - Last step gets `reward = success_reward`; others keep original reward
6. Return relabeled list

**Constants:**

```python
ZONE_POSITIONS  # shape (3, 3) -- xyz for zone_A, zone_B, zone_C
ZONE_GOAL_IDS   # shape (3, 3) -- eye(3), one-hot per zone
```

---

### policy/config.py

**`P3Config` (dataclass)**

All P3 hyperparameters. Pass to `P3Trainer`.

Key fields:

| Field | Default | Description |
|-------|---------|-------------|
| `feat_dim` | 1536 | RSSM feat dim -- must match P2's world model |
| `action_dim` | 6 | Action space dimension |
| `actor_lr` / `critic_lr` | 3e-5 | Learning rates (from DreamerV3 paper) |
| `gamma` | 0.997 | Discount factor |
| `lambda_` | 0.95 | TD(lambda) mixing coefficient |
| `imagination_horizon` | 15 | Steps to imagine per update |
| `prefill_steps` | 2000 | Random actions before learning starts |
| `batch_size` | 16 | Batch size for WM and AC training |
| `buffer_capacity` | 100_000 | Max transitions in EpisodeBuffer |
| `her_enabled` | True | Toggle Visual HER |
| `her_ratio` | 0.5 | Fraction of episodes relabeled |
| `slow_critic_update_freq` | 100 | Steps between EMA slow-target syncs |
| `device` | "cuda:0" | Target device |

---

### policy/actor_critic.py

#### `Actor`

```python
Actor(feat_dim=1536, action_dim=6, hidden=[512,256],
      min_std=0.1, max_std=1.0, entropy_scale=3e-4)
```

MLP with LayerNorm after each hidden layer. Outputs a **TanhNormal** distribution over actions.

- `forward(feat) -> (action, log_prob)` -- sampled action + log-prob sum
- `mean_action(feat) -> action` -- deterministic eval action (tanh of mean)
- `entropy(feat) -> (B,)` -- base Normal entropy (approximation; TanhTransform has no closed-form)
- `loss(imagine_feat, lambda_returns) -> scalar` -- REINFORCE + entropy bonus

**Loss formula:**

```
L_actor = -E[returns * log_prob] - entropy_scale * E[entropy]
```

#### `Critic`

```python
Critic(feat_dim=1536, hidden=[512,256],
       slow_target_freq=100, slow_frac=0.02)
```

Value MLP + slow EMA copy (no grad) for stable bootstrap targets.

- `forward(feat) -> (B, 1)` -- main value
- `slow_value(feat) -> (B, 1)` -- EMA target (no grad)
- `update_slow_target()` -- call once per update step; EMA syncs every `slow_target_freq` steps
- `loss(imagine_feat, lambda_returns) -> scalar` -- MSE(V_main, returns.detach())

**EMA update:**

```python
slow_p.data.lerp_(main_p.data, slow_frac)  # alpha=0.02 toward main
```

#### `lambda_return`

```python
lambda_return(rewards, values, dones, gamma=0.997, lambda_=0.95) -> Tensor
```

Computes TD(lambda) targets via backward recursion:

```
V_lambda(t) = r(t) + gamma*(1-d(t)) * [(1-lambda)*V(t+1) + lambda*V_lambda(t+1)]
V_lambda(H) = V(H)  (boundary bootstrap)
```

- `rewards`: `(H, B)` imagined rewards
- `values`: `(H+1, B)` slow-target critic values (include bootstrap at index H)
- `dones`: `(H, B)` float continuation flags (1=not done, 0=done/terminal)
- Returns: `(H, B)` lambda-return targets

---

### policy/train_loop.py

#### `WorldModelInterface` (ABC)

P2 must implement this interface. P3 calls only these methods:

```python
class WorldModelInterface(abc.ABC):
    def encode_obs(self, obs: dict, device: str) -> Tensor   # (B, feat_dim)
    def imagine_step(self, feat, action) -> (next_feat, pred_reward, pred_cont)
    def train_batch(self, batch: Batch, device: str) -> dict[str, float]
    def get_feat_dim(self) -> int
    def get_initial_feat(self, batch_size, device) -> Tensor  # zeros default
```

**Contract for P2:**

- `feat_dim` returned by `get_feat_dim()` must equal `P3Config.feat_dim` (1536)
- `encode_obs` receives a dict with v2 obs keys, numpy arrays, may be batched `(B,...)`
- `imagine_step` receives tensors on `device`; returns tensors on same device
- `pred_cont`: 1.0 = episode continues, 0.0 = terminal

#### `P3Trainer`

```python
P3Trainer(env, world_model: WorldModelInterface, cfg: P3Config = P3Config())
```

**Public methods:**

| Method | Description |
|--------|-------------|
| `run(total_steps)` | Main loop: collect -> buffer -> WM train -> AC train -> log |
| `eval_episode() -> dict` | One deterministic eval episode; returns reward/length/success |
| `save(path=None) -> Path` | Save checkpoint: actor, critic, optimizers, global_step |
| `load(path)` | Restore checkpoint |

**Training step (per collected env step):**

1. `_select_action()` -- random uniform during prefill; Actor forward after
2. `env.step(action)` -> add to buffer
3. If `len(buffer) >= prefill_steps`:
   - `_train_world_model()` -> `buffer.sample() -> wm.train_batch()`
   - `_train_actor_critic()`:
     - Encode start feats: `wm.encode_obs(batch.obs)`
     - Imagine H steps (actor selects actions, WM features detached)
     - `lambda_return(rewards, slow_values, conts)`
     - Critic update: MSE loss -> `critic_opt.step()`
     - Actor update: re-imagine with grad -> REINFORCE + entropy -> `actor_opt.step()`

---

### scripts/train_p3.py

Entry point. **Owns `AppLauncher`** -- never import AppLauncher elsewhere.

```bash
# Normal run (requires P1 obs_v2 + P2 WorldModel)
python scripts/train_p3.py --headless

# With GPU selection and custom steps
python scripts/train_p3.py --headless --num_envs 1 --steps 200000 --seed 0

# Dev mode -- no Isaac, no P2 (pure P3 loop test)
python scripts/train_p3.py --headless --mock_wm

# Disable W&B logging
python scripts/train_p3.py --headless --wandb_mode disabled
```

`--mock_wm` uses `_MockWorldModel` (random RSSM features). For testing P3 loop isolation only -- not for real experiments.

---

## Integration Points

### P1 (Environment)

P3 expects P1 to provide:

```python
from env.warehouse_env import WarehouseEnvCfg, WarehouseGymEnv
```

With **obs_v2** (9 keys above) and **action space `(6,)`**.

> **Current status (2026-06-16):** `env/warehouse_env.py` still uses old action `(2,)` and 5 obs keys. P1 must commit obs_v2 changes before `train_p3.py` works without `--mock_wm`.

### P2 (World Model)

P3 expects P2 to expose:

```python
from models.dreamerv3 import WarehouseWorldModel
```

Where `WarehouseWorldModel` implements `WorldModelInterface` with:
- `get_feat_dim() -> 1536`  (must match P3Config.feat_dim)
- RSSM built from `vendor/configs.yaml`: `dyn_deter=512, dyn_stoch=32, dyn_discrete=32`
- feat_dim = 512 + 32*32 = 1536

> **If P2 is not ready:** Run with `--mock_wm`. `_MockWorldModel` has the same interface.

**RSSM feature dimension auto-check in P3Trainer:**

```python
# P3Trainer.__init__ raises early if mismatch:
if world_model.get_feat_dim() != cfg.feat_dim:
    raise ValueError(f"WorldModel.get_feat_dim()={feat_dim} != P3Config.feat_dim={cfg.feat_dim}")
```

### P5 (Logger & Seed)

P3 uses P5's utilities:

```python
from training.logger import Logger    # W&B wrapper with stdout fallback
from training.seed import seed_everything
```

P3 does **not** use `training/replay_buffer.py` (P5-owned, old contract). P3 has its own `buffer/replay_buffer.py`.

---

## Running the Training Loop

### Prerequisites

1. P1 environment must expose obs_v2 + action (6,) in `env/warehouse_env.py`
2. P2 world model must expose `WarehouseWorldModel` in `models/dreamerv3/__init__.py`
3. Isaac Lab installed and sourced
4. CUDA GPU available (or set `P3Config(device="cpu")` for CPU debug)

### Steps

```bash
# 1. Source Isaac Lab (adjust path to your install)
source /path/to/isaaclab/_isaac_sim/setup_conda_env.sh

# 2. Run training
python scripts/train_p3.py --headless --num_envs 1 --steps 200000

# 3. Checkpoint saved automatically on exit:
#    training/results/p3/p3_step200000.pt
```

### Dev / Isolated P3 Testing (no Isaac Lab needed)

```bash
# Tests buffer + actor-critic logic, no simulator required
python scripts/train_p3.py --headless --mock_wm

# Or run unit tests directly
pytest tests/test_p3_buffer.py tests/test_p3_actor_critic.py tests/test_p3_visual_her.py -v
```

---

## Testing

### Test Files

| File | What it tests |
|------|---------------|
| `tests/test_p3_buffer.py` | `EpisodeBuffer` add/sample/ring-overwrite, HER integration (9 tests) |
| `tests/test_p3_actor_critic.py` | `Actor` shapes/bounds/loss, `Critic` forward/slow-target, `lambda_return` shapes/edge-cases (13 tests) |
| `tests/test_p3_visual_her.py` | `make_visual_her_fn` -- no-grasp, her_ratio, goal correctness, nearest-zone, post-grasp range (12 tests) |

### Running Tests

```bash
# All P3 tests
pytest tests/test_p3_buffer.py tests/test_p3_actor_critic.py tests/test_p3_visual_her.py -v

# With coverage report
pytest tests/test_p3_*.py --cov=buffer --cov=policy --cov-report=term-missing
```

Tests use `FEAT_DIM=64` (prod=1536) and `(3,8,8)` pixels (prod=`(3,64,64)`) for CPU speed.

---

## Key Hyperparameters

All in `policy/config.py -> P3Config`.

```python
# Quick CPU debug override
cfg = P3Config(
    feat_dim=1536,
    prefill_steps=100,
    batch_size=4,
    imagination_horizon=5,
    buffer_capacity=1000,
    device="cpu",
    wandb_mode="disabled",
)
trainer = P3Trainer(env, world_model, cfg)
trainer.run(total_steps=500)
```

**Tuning guide:**

- `imagination_horizon`: Higher = better credit assignment, slower training. Start at 15.
- `lambda_`: 0.95 standard. Lower means more bootstrap bias, less variance.
- `actor_entropy_scale`: Increase if policy collapses early; decrease if it won't commit.
- `her_ratio`: 0.5 means 50% of completed episodes are relabeled. Increase for sparse reward.
- `slow_critic_update_freq`: Higher = more stable but slower bootstrap adaptation.

---

## Common Errors

### `ValueError: WorldModel.get_feat_dim()=X != P3Config.feat_dim=1536`

P2's RSSM config doesn't match. Check `models/dreamerv3/vendor/configs.yaml`:

```yaml
dyn_deter: 512
dyn_stoch: 32
dyn_discrete: 32
# feat_dim = 512 + 32*32 = 1536
```

### `ImportError: P2's WarehouseWorldModel not found`

P2 hasn't committed yet. Run with `--mock_wm` until P2 is ready.

### `RuntimeError: EpisodeBuffer is empty`

Training started before buffer has `prefill_steps` transitions. The trainer guards this with `len(self.buffer) >= self.cfg.prefill_steps` -- should not occur in normal usage.

### `KeyError: 'holding'` in Visual HER

Obs dict missing `holding` key -- P1 hasn't committed obs_v2 yet. Visual HER uses `holding` to detect grasp events.

### `RuntimeError: AppLauncher already initialized`

Only `scripts/train_p3.py` may instantiate `AppLauncher`. Any `from isaaclab.app import AppLauncher` in non-entry files causes double-launch crash.

### Action space mismatch `(2,)` vs `(6,)`

`env/warehouse_env.py` still uses old action space. P1 must update to obs_v2 + action(6,) before full loop runs.

---

## Architecture Notes

### Why imagination-based training?

DreamerV3's actor-critic trains on imagined RSSM rollouts, not real env rollouts. Benefits:
- Decouples sample collection from policy learning
- Allows many gradient updates per env step (`train_ratio`)
- Critic's slow-target prevents target explosion in long imagination horizons

### Why Visual HER instead of standard HER?

Standard HER relabels `goal` to the final state reached. Visual HER additionally:
- Detects whether the robot **grasped** a box (via `holding >= 0.5`)
- Identifies the **nearest delivery zone** post-grasp (2D distance)
- Relabels both `goal` (xyz) AND `goal_id` (category one-hot)

This gives reward signal for episodes where the robot grasped the correct box category but delivered it to the wrong zone -- a common failure mode in the warehouse task.

### Old vs New Buffer

| File | Owner | Contract | Episode tracking |
|------|-------|----------|-----------------|
| `training/replay_buffer.py` | P5 | Old obs keys (5) | No |
| `buffer/replay_buffer.py` | P3 | v2 obs keys (9) | Yes, with HER |

Do not confuse these two files.
