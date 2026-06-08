# Env Pickup Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the warehouse env from single-goal navigation to pick→carry→place: 6-dim action with Franka arm IK, observation contract v2 (`goal_id` + manipulation keys), ~18 reachable boxes, and staged grasp/place reward.

**Architecture:** Keep the manager-based `WarehouseRLEnv` + `WarehouseGymEnv` split. Add an arm IK action term (`DifferentialInverseKinematicsActionCfg`) and a binary gripper term alongside the existing base-velocity term; the Gym wrapper maps the external `(6,)` policy action onto the internal `(7,)` joint action. Grasp is made reliable by a kinematic attach on grasp-success (physics grip is too fragile for early RL). Pure-tensor logic (action split, grasp detection, staged reward, curriculum) lives in small focused modules with pytest unit tests; Isaac-runtime wiring is verified through `tests/test_env.py`.

**Tech Stack:** Isaac Lab 5.1, PyTorch, Gymnasium, pytest. Reference: `Isaac-Lift-Cube-Franka-v0` for the IK action term + lift reward pattern.

---

## File Structure

**New files:**
- `env/action_pickup.py` — pure-tensor split of the external `(6,)` action into `(base2, ee_delta3, gripper1)`. Unit-testable, no Isaac import.
- `env/grasp.py` — pure-tensor grasp-success / drop detection given ee/box/gripper tensors. Unit-testable.
- `env/curriculum.py` — pure-tensor `goal_id` one-hot + goal-xyz anneal helpers. Unit-testable.
- `tests/test_action_pickup.py`, `tests/test_grasp.py`, `tests/test_curriculum.py`, `tests/test_reward_pickup.py` — pure pytest (no GPU).

**Modified files:**
- `env/warehouse_scene.py` — reduce 54→18 boxes, place reachable; add `TARGET_BOX_SPECS`.
- `env/warehouse_reward.py` — add staged reward + pickup termination fns (read env attrs).
- `env/warehouse_env.py` — IK + gripper action terms; obs v2 terms; `goal_id`/`box_pos`/`holding` buffers; grasp+attach in step; `(6,)`/Dict spaces.
- `tests/test_env.py` — interface-contract assertions for v2 obs/action.
- `configs/env_config.yaml` — sync to pickup (count 18, action 6, staged reward).

---

## Task 1: Reachable target boxes in the scene

**Files:**
- Modify: `env/warehouse_scene.py` (ITEM_SPECS block `:193-211`, `__post_init__` `:414-429`)
- Test: `tests/test_layout_grid.py` (append) — pure, no GPU

Reduce from 54 shelf boxes to 18 floor-level **target** boxes (one per rack, category cycles per rack), each placed on the floor directly in front of its rack so the Franka (reach ~0.85m, mounted ~0.4m up) can grasp from a feasible base pose. Keep the 18 racks + shelf decks as the navigation maze; remove the 54 shelf boxes.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_layout_grid.py`:

```python
def test_target_box_specs_count_and_reachability():
    """18 target boxes, one per rack, on the floor and within Franka reach of rack front."""
    from env.warehouse_scene import TARGET_BOX_SPECS, RACK_POSITIONS

    assert len(TARGET_BOX_SPECS) == len(RACK_POSITIONS) == 18
    for name, size, mass, pos in TARGET_BOX_SPECS:
        # box rests on the floor: center z == half the cube height
        assert abs(pos[2] - size / 2.0) < 1e-6, f"{name} not floor-resting (z={pos[2]})"
        # each box sits within 0.85 m (Franka reach) of some rack front in the xy-plane
        near = min(((pos[0] - rx) ** 2 + (pos[1] - ry) ** 2) ** 0.5 for rx, ry, _ in RACK_POSITIONS)
        assert near <= 0.85, f"{name} unreachable (nearest rack {near:.2f} m)"

def test_target_box_categories_cycle():
    """Categories cycle fragile/regular/heavy across the 18 racks."""
    from env.warehouse_scene import TARGET_BOX_SPECS
    cats = [name.split("_")[0] for name, *_ in TARGET_BOX_SPECS]
    assert cats.count("fragile") == cats.count("regular") == cats.count("heavy") == 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_layout_grid.py::test_target_box_specs_count_and_reachability -v`
Expected: FAIL with `ImportError: cannot import name 'TARGET_BOX_SPECS'`

- [ ] **Step 3: Write minimal implementation**

In `env/warehouse_scene.py`, replace the `ITEM_SPECS` construction block (`:201-211`) with an 18-box floor layout. Place each box 0.5 m toward room-center-y from its rack (in front of the shelf, on the floor):

```python
# 18 target boxes: one per rack, on the FLOOR in front of the rack (within Franka reach).
# Category cycles fragile/regular/heavy by rack index → 6 of each across 18 racks.
# Boxes are graspable rigid bodies; the commanded box is selected at runtime by goal_id.
_BOX_CATS   = ("fragile", "regular", "heavy")
_BOX_SIZES  = (BOX_SMALL_SIZE, BOX_MED_SIZE, BOX_LARGE_SIZE)
_BOX_MASSES = BOX_MASSES
BOX_FRONT_OFFSET = 0.5  # meters toward room center (−y) from rack center, on the floor

TARGET_BOX_SPECS: list[tuple[str, float, float, tuple[float, float, float]]] = []
_cat_cnt = {"fragile": 0, "regular": 0, "heavy": 0}
for _i, (_rx, _ry, _) in enumerate(RACK_POSITIONS):
    _ci   = _i % 3
    _cat  = _BOX_CATS[_ci]
    _size = _BOX_SIZES[_ci]
    _mass = _BOX_MASSES[_ci]
    _name = f"{_cat}_{_cat_cnt[_cat]}"
    _cat_cnt[_cat] += 1
    # front of rack (toward shipping/−y), resting on the floor (center z = size/2)
    _bx, _by, _bz = _rx, _ry - BOX_FRONT_OFFSET, _size / 2.0
    TARGET_BOX_SPECS.append((_name, _size, _mass, (_bx, _by, _bz)))

# Back-compat alias: env code iterates ITEM_SPECS.
ITEM_SPECS = TARGET_BOX_SPECS
```

Then in `__post_init__` the `for name, size, mass, pos in ITEM_SPECS` loop at `:426-427` already spawns whatever is in `ITEM_SPECS`, so it now spawns 18 floor boxes — no change needed there. Keep the rack + shelf-deck loop. Update the docstring counts (`:414-421`) from 54 to 18.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_layout_grid.py -v`
Expected: PASS (all, including the two new tests)

- [ ] **Step 5: Commit**

```bash
git add env/warehouse_scene.py tests/test_layout_grid.py
git commit -m "feat(env): 18 floor-level reachable target boxes (was 54 shelf boxes)"
```

---

## Task 2: Pure-tensor action split (6 → base2 + ee3 + grip1)

**Files:**
- Create: `env/action_pickup.py`
- Test: `tests/test_action_pickup.py`

The external policy action is `(N, 6) = [base_lin, base_ang, ee_dx, ee_dy, ee_dz, gripper]`. Split into the three groups the wrapper forwards. Scale the EE delta by a per-step reach so the policy's `[-1,1]` maps to a sane Cartesian step.

- [ ] **Step 1: Write the failing test**

`tests/test_action_pickup.py`:

```python
import torch
from env.action_pickup import split_action, EE_STEP_M

def test_split_shapes():
    a = torch.zeros(4, 6)
    base, ee, grip = split_action(a)
    assert base.shape == (4, 2)
    assert ee.shape == (4, 3)
    assert grip.shape == (4, 1)

def test_ee_delta_scaled_by_step():
    a = torch.zeros(1, 6)
    a[0, 2] = 1.0  # full +x ee command
    _, ee, _ = split_action(a)
    assert torch.allclose(ee[0], torch.tensor([EE_STEP_M, 0.0, 0.0]))

def test_passthrough_base_and_gripper():
    a = torch.tensor([[0.3, -0.7, 0.0, 0.0, 0.0, 0.9]])
    base, _, grip = split_action(a)
    assert torch.allclose(base[0], torch.tensor([0.3, -0.7]))
    assert torch.allclose(grip[0], torch.tensor([0.9]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_action_pickup.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'env.action_pickup'`

- [ ] **Step 3: Write minimal implementation**

`env/action_pickup.py`:

```python
"""Pure-tensor split of the external (N,6) pickup action. No Isaac import (unit-testable)."""

from __future__ import annotations

import torch

EE_STEP_M = 0.05  # meters of EE travel commanded per control step at action == 1.0


def split_action(action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split (N,6) [base_lin, base_ang, ee_dx, ee_dy, ee_dz, gripper] into (base2, ee3, grip1).

    The EE delta is scaled from [-1,1] to a Cartesian step (EE_STEP_M); base and gripper pass through.
    """
    base = action[:, 0:2]
    ee   = action[:, 2:5] * EE_STEP_M
    grip = action[:, 5:6]
    return base, ee, grip
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_action_pickup.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add env/action_pickup.py tests/test_action_pickup.py
git commit -m "feat(env): pure-tensor 6-dim pickup action split"
```

---

## Task 3: Pure-tensor grasp-success / drop detection

**Files:**
- Create: `env/grasp.py`
- Test: `tests/test_grasp.py`

Grasp-success = gripper commanded closed AND end-effector within `grip_radius` of the target box AND box lifted above its resting height. Drop = was holding, now the EE has separated from the box. Pure tensors so it unit-tests without Isaac.

- [ ] **Step 1: Write the failing test**

`tests/test_grasp.py`:

```python
import torch
from env.grasp import grasp_success, grasp_lost, GRIP_RADIUS_M, LIFT_M

def _ee():  return torch.tensor([[0.0, 0.0, 0.30]])
def _box(): return torch.tensor([[0.0, 0.0, 0.30]])  # box at ee

def test_grasp_success_when_close_closed_and_lifted():
    ok = grasp_success(
        ee_pos=_ee(), box_pos=_box(),
        gripper_closed=torch.tensor([True]),
        box_lift=torch.tensor([LIFT_M + 0.01]),
    )
    assert bool(ok[0]) is True

def test_no_grasp_when_far():
    far_box = torch.tensor([[1.0, 0.0, 0.30]])
    ok = grasp_success(
        ee_pos=_ee(), box_pos=far_box,
        gripper_closed=torch.tensor([True]),
        box_lift=torch.tensor([LIFT_M + 0.01]),
    )
    assert bool(ok[0]) is False

def test_no_grasp_when_open():
    ok = grasp_success(
        ee_pos=_ee(), box_pos=_box(),
        gripper_closed=torch.tensor([False]),
        box_lift=torch.tensor([LIFT_M + 0.01]),
    )
    assert bool(ok[0]) is False

def test_grasp_lost_when_holding_and_ee_leaves_box():
    lost = grasp_lost(
        holding=torch.tensor([True]),
        ee_pos=_ee(), box_pos=torch.tensor([[0.5, 0.0, 0.30]]),
    )
    assert bool(lost[0]) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_grasp.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'env.grasp'`

- [ ] **Step 3: Write minimal implementation**

`env/grasp.py`:

```python
"""Pure-tensor grasp-success / grasp-lost detection. No Isaac import (unit-testable)."""

from __future__ import annotations

import torch

GRIP_RADIUS_M = 0.08  # EE must be within this of the box center to count as grasping
LIFT_M        = 0.05  # box must rise this far above its resting height to count as lifted


def grasp_success(
    ee_pos: torch.Tensor,
    box_pos: torch.Tensor,
    gripper_closed: torch.Tensor,
    box_lift: torch.Tensor,
) -> torch.Tensor:
    """(N,) bool: gripper closed AND EE within GRIP_RADIUS_M of box AND box lifted > LIFT_M."""
    near = torch.norm(ee_pos - box_pos, dim=-1) < GRIP_RADIUS_M
    lifted = box_lift > LIFT_M
    return near & gripper_closed & lifted


def grasp_lost(
    holding: torch.Tensor,
    ee_pos: torch.Tensor,
    box_pos: torch.Tensor,
) -> torch.Tensor:
    """(N,) bool: currently holding but the EE has separated from the box (> 2×GRIP_RADIUS_M)."""
    separated = torch.norm(ee_pos - box_pos, dim=-1) > (2.0 * GRIP_RADIUS_M)
    return holding & separated
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_grasp.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add env/grasp.py tests/test_grasp.py
git commit -m "feat(env): pure-tensor grasp-success/lost detection"
```

---

## Task 4: Pure-tensor curriculum helpers (goal_id one-hot + goal anneal)

**Files:**
- Create: `env/curriculum.py`
- Test: `tests/test_curriculum.py`

`goal_id` is a one-hot over the 3 categories, derived from the per-env commanded category index. `goal` xyz is multiplied by an anneal factor `alpha` (1.0 → 0.0 across curriculum) so the policy weans off the leaked zone location. `box_pos` is never annealed (handled in env, not here).

- [ ] **Step 1: Write the failing test**

`tests/test_curriculum.py`:

```python
import torch
from env.curriculum import goal_id_onehot, anneal_goal

def test_goal_id_onehot():
    idx = torch.tensor([0, 2, 1])
    oh = goal_id_onehot(idx, num_cats=3)
    assert oh.shape == (3, 3)
    assert torch.allclose(oh[0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(oh[1], torch.tensor([0.0, 0.0, 1.0]))

def test_anneal_full_then_zero():
    goal = torch.tensor([[3.0, -12.0, 0.0]])
    assert torch.allclose(anneal_goal(goal, alpha=1.0), goal)
    assert torch.allclose(anneal_goal(goal, alpha=0.0), torch.zeros_like(goal))
    assert torch.allclose(anneal_goal(goal, alpha=0.5), goal * 0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_curriculum.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'env.curriculum'`

- [ ] **Step 3: Write minimal implementation**

`env/curriculum.py`:

```python
"""Pure-tensor curriculum helpers: goal_id one-hot + goal-xyz anneal. No Isaac import."""

from __future__ import annotations

import torch


def goal_id_onehot(cat_idx: torch.Tensor, num_cats: int = 3) -> torch.Tensor:
    """(N,num_cats) float one-hot of the commanded category index (0=fragile,1=regular,2=heavy)."""
    return torch.nn.functional.one_hot(cat_idx.long(), num_classes=num_cats).float()


def anneal_goal(goal_xyz: torch.Tensor, alpha: float) -> torch.Tensor:
    """Scale goal xyz by alpha (1.0 = full leak, 0.0 = hidden). box_pos is annealed elsewhere (never)."""
    return goal_xyz * alpha
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_curriculum.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add env/curriculum.py tests/test_curriculum.py
git commit -m "feat(env): pure-tensor curriculum helpers (goal_id, anneal)"
```

---

## Task 5: Staged reward + pickup termination functions

**Files:**
- Modify: `env/warehouse_reward.py` (append new fns; keep existing nav fns for reference/baseline)
- Test: `tests/test_reward_pickup.py`

New reward terms read runtime buffers the env will populate in Task 8: `env.ee_pos (N,3)`, `env.box_pos (N,3)`, `env.holding (N,) bool`, `env.goal_pos (N,3)`, plus `env.grasp_event (N,) bool` and `env.drop_event (N,) bool` (one-step flags set in `step`). Functions are pure tensor given those attrs, so they unit-test against a `SimpleNamespace` fake env.

- [ ] **Step 1: Write the failing test**

`tests/test_reward_pickup.py`:

```python
import torch
from types import SimpleNamespace
from env.warehouse_reward import (
    approach_box_distance, carry_distance, grasp_success_reward,
    drop_penalty, pickup_delivered, pickup_delivered_reward,
)

def _env(**kw):
    base = dict(
        num_envs=1, device="cpu",
        ee_pos=torch.tensor([[0.0, 0.0, 0.3]]),
        box_pos=torch.tensor([[0.0, 0.0, 0.3]]),
        holding=torch.tensor([False]),
        goal_pos=torch.tensor([[0.0, -12.0, 0.0]]),
        grasp_event=torch.tensor([False]),
        drop_event=torch.tensor([False]),
    )
    base.update(kw)
    return SimpleNamespace(**base)

def test_approach_distance_zero_when_ee_on_box():
    assert torch.allclose(approach_box_distance(_env()), torch.tensor([0.0]))

def test_approach_gated_off_when_holding():
    e = _env(holding=torch.tensor([True]), ee_pos=torch.tensor([[1.0, 0.0, 0.3]]))
    assert torch.allclose(approach_box_distance(e), torch.tensor([0.0]))  # gated → 0

def test_carry_distance_active_only_when_holding():
    e = _env(holding=torch.tensor([True]),
             box_pos=torch.tensor([[0.0, 0.0, 0.3]]),
             goal_pos=torch.tensor([[0.0, -3.0, 0.0]]))
    assert torch.allclose(carry_distance(e), torch.tensor([3.0]))
    assert torch.allclose(carry_distance(_env()), torch.tensor([0.0]))  # not holding → 0

def test_grasp_reward_fires_on_event():
    assert torch.allclose(grasp_success_reward(_env(grasp_event=torch.tensor([True]))),
                          torch.tensor([1.0]))

def test_drop_penalty_fires_on_event():
    assert torch.allclose(drop_penalty(_env(drop_event=torch.tensor([True]))),
                          torch.tensor([1.0]))

def test_pickup_delivered_requires_holding_and_in_zone():
    e = _env(holding=torch.tensor([True]),
             box_pos=torch.tensor([[0.0, -12.0, 0.3]]),
             goal_pos=torch.tensor([[0.0, -12.0, 0.0]]))
    assert bool(pickup_delivered(e)[0]) is True
    assert torch.allclose(pickup_delivered_reward(e), torch.tensor([1.0]))
    assert bool(pickup_delivered(_env())[0]) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_reward_pickup.py -v`
Expected: FAIL with `ImportError: cannot import name 'approach_box_distance'`

- [ ] **Step 3: Write minimal implementation**

Append to `env/warehouse_reward.py`:

```python
# ── Pickup (staged) reward + termination ──────────────────────────────
# These read runtime buffers populated by WarehouseRLEnv/WarehouseGymEnv each step:
#   env.ee_pos (N,3), env.box_pos (N,3), env.holding (N,) bool, env.goal_pos (N,3),
#   env.grasp_event (N,) bool, env.drop_event (N,) bool.
DELIVER_RADIUS_M = 1.5  # box within this xy-distance of the goal zone center = delivered


def approach_box_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Phase A dense: distance(ee, box), zero while holding (use with negative weight)."""
    d = torch.norm(env.ee_pos - env.box_pos, dim=-1)
    return torch.where(env.holding, torch.zeros_like(d), d)


def carry_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Phase B dense: xy-distance(box, goal), zero while NOT holding (use with negative weight)."""
    d = torch.norm(env.box_pos[:, :2] - env.goal_pos[:, :2], dim=-1)
    return torch.where(env.holding, d, torch.zeros_like(d))


def grasp_success_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """+1 on the step grasp succeeds (one-shot). Use with positive weight (e.g. 5.0)."""
    return env.grasp_event.float()


def drop_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """+1 on the step the box is dropped outside a zone (one-shot). Use with negative weight."""
    return env.drop_event.float()


def pickup_delivered(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(N,) bool: holding AND box xy within DELIVER_RADIUS_M of the goal zone center."""
    in_zone = torch.norm(env.box_pos[:, :2] - env.goal_pos[:, :2], dim=-1) < DELIVER_RADIUS_M
    return env.holding & in_zone


def pickup_delivered_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """+1 per step while the held box is delivered in its zone (float of pickup_delivered)."""
    return pickup_delivered(env).float()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_reward_pickup.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add env/warehouse_reward.py tests/test_reward_pickup.py
git commit -m "feat(env): staged pickup reward + delivery termination fns"
```

---

## Task 6: Arm IK + gripper action terms (Isaac wiring)

**Files:**
- Modify: `env/warehouse_env.py` `ActionsCfg` (`:125-145`)

Add a differential IK action term for the arm (relative position, fixed top-down orientation → 3 dims) and a binary gripper term (1 dim) after the existing base-velocity term (3 dims). Internal action vector becomes `(7,)` in declaration order: base(3) + arm_ik(3) + gripper(1).

- [ ] **Step 1: Add the action terms**

In `env/warehouse_env.py`, add imports near the other mdp import (`:28`):

```python
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions import DifferentialInverseKinematicsActionCfg, BinaryJointPositionActionCfg
```

Extend `ActionsCfg` (after `base_vel`):

```python
    # Arm: differential IK on the 7 panda joints. Relative-mode position command (3 dims:
    # dx,dy,dz in base frame); orientation held fixed (top-down) by the controller. Body
    # "panda_hand" is the Franka EE link. Copy of the Isaac-Lift-Cube-Franka-v0 IK term.
    arm_ik = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(
            command_type="position", use_relative_mode=True, ik_method="dls"
        ),
        scale=1.0,
    )

    # Gripper: open/close as a binary action mapped to the two finger joints.
    gripper = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger_joint.*"],
        open_command_expr={"panda_finger_joint.*": 0.035},
        close_command_expr={"panda_finger_joint.*": 0.0},
    )
```

- [ ] **Step 2: Verify the cfg imports and the term count is right**

Run (no GPU needed — config import only):
`python -c "from env.warehouse_env import ActionsCfg; c=ActionsCfg(); print([t for t in vars(c) if not t.startswith('_')])"`
Expected: prints a list containing `base_vel`, `arm_ik`, `gripper`.

If `ImportError` on the action classes, find the correct path:
`python -c "import isaaclab.envs.mdp as m; print([x for x in dir(m) if 'Action' in x])"`
and adjust the import to the names listed.

- [ ] **Step 3: Commit**

```bash
git add env/warehouse_env.py
git commit -m "feat(env): add Franka arm IK + binary gripper action terms"
```

---

## Task 7: Observation v2 terms (Isaac wiring)

**Files:**
- Modify: `env/warehouse_env.py` obs functions (`:89-99`, `:148-168`)

Replace `goal_emb` with `goal_id` and add manipulation obs functions. `ee_pos` is the `panda_hand` body position in the base frame; `gripper` is normalized finger opening; `holding` reads `env.holding`; `box_pos` reads `env.box_pos` (commanded target, env-local). These buffers are created in Task 8.

- [ ] **Step 1: Add obs functions**

In `env/warehouse_env.py`, replace `goal_embedding` (`:96-98`) and add new fns:

```python
def goal_id(env: ManagerBasedRLEnv) -> torch.Tensor:
    """One-hot (num_envs, 3) commanded category [fragile, regular, heavy]. Reads env.goal_id_buf."""
    if hasattr(env, "goal_id_buf"):
        return env.goal_id_buf
    return torch.zeros(env.num_envs, 3, device=env.device)


def ee_position(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """End-effector (panda_hand) xyz in the base frame, shape (num_envs, 3)."""
    robot: Articulation = env.scene[asset_cfg.name]
    ee = robot.body_names.index("panda_hand")
    base = robot.body_names.index("base_link")
    return robot.data.body_pos_w[:, ee] - robot.data.body_pos_w[:, base]


def gripper_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Normalized finger opening (num_envs, 1) in [0,1] (0=closed, 1=open at 0.035 m)."""
    robot: Articulation = env.scene[asset_cfg.name]
    j = robot.joint_names.index("panda_finger_joint1")
    return (robot.data.joint_pos[:, j:j+1] / 0.035).clamp(0.0, 1.0)


def holding_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(num_envs, 1) float: 1.0 if the target box is currently grasped."""
    if hasattr(env, "holding"):
        return env.holding.float().unsqueeze(-1)
    return torch.zeros(env.num_envs, 1, device=env.device)


def box_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Target box xyz, env-local (num_envs, 3). Reads env.box_pos (commanded by goal_id)."""
    if hasattr(env, "box_pos"):
        return env.box_pos
    return torch.zeros(env.num_envs, 3, device=env.device)
```

- [ ] **Step 2: Rewire PolicyCfg**

Replace the `PolicyCfg` body (`:153-166`):

```python
    @configclass
    class PolicyCfg(ObsGroup):
        """Policy obs group (v2): nav + manipulation, separate keys (no concat)."""

        pixels    = ObsTerm(func=camera_rgb)
        position  = ObsTerm(func=robot_position)
        heading   = ObsTerm(func=robot_heading)
        goal      = ObsTerm(func=goal_position)
        goal_id   = ObsTerm(func=goal_id)
        ee_pos    = ObsTerm(func=ee_position)
        gripper   = ObsTerm(func=gripper_state)
        holding   = ObsTerm(func=holding_state)
        box_pos   = ObsTerm(func=box_position)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False
```

- [ ] **Step 3: Verify config imports**

Run: `python -c "from env.warehouse_env import ObservationsCfg; print('ok')"`
Expected: prints `ok` (no GPU needed).

- [ ] **Step 4: Commit**

```bash
git add env/warehouse_env.py
git commit -m "feat(env): observation contract v2 (goal_id + manipulation keys)"
```

---

## Task 8: Env runtime buffers, action mapping, grasp + kinematic attach

**Files:**
- Modify: `env/warehouse_env.py` — `WarehouseRLEnv` (`:245-320`), `WarehouseGymEnv` (`:324-432`)
- Modify: `env/warehouse_env.py` — `RewardsCfg`/`TerminationsCfg`/`__post_init__` + reward imports (`:41-48`, `:188-212`, `:235`)

Add per-env runtime buffers (`goal_id_buf`, `box_pos`, `holding`, `grasp_event`, `drop_event`, `ee_pos`, `box_cat_idx`, `target_box_name`) and wire them. On reset: pick a commanded category, select that category's box as the target, set `goal_pos` to the matching zone. Each step: refresh `ee_pos`/`box_pos`, evaluate grasp-success → set `holding` and attach the box to the EE (kinematic carry), evaluate drop → detach. Map the external `(6,)` action to the internal `(7,)` joint action.

- [ ] **Step 1: Add buffers + target selection to `WarehouseRLEnv.__init__`**

After `self._resample_goals(...)` (`:257`), allocate buffers and per-category box lists. `TARGET_BOX_SPECS` names are `fragile_k`/`regular_k`/`heavy_k`:

```python
        from env.warehouse_scene import TARGET_BOX_SPECS
        self._cat_names = ("fragile", "regular", "heavy")
        self._boxes_by_cat = {
            c: [n for (n, *_ ) in TARGET_BOX_SPECS if n.startswith(c)] for c in self._cat_names
        }
        N, dev = self.num_envs, self.device
        self.goal_id_buf = torch.zeros(N, 3, device=dev)
        self.box_cat_idx = torch.zeros(N, dtype=torch.long, device=dev)
        self.target_box_name = ["" for _ in range(N)]
        self.box_pos = torch.zeros(N, 3, device=dev)
        self.ee_pos = torch.zeros(N, 3, device=dev)
        self.holding = torch.zeros(N, dtype=torch.bool, device=dev)
        self.grasp_event = torch.zeros(N, dtype=torch.bool, device=dev)
        self.drop_event = torch.zeros(N, dtype=torch.bool, device=dev)
        self._sample_targets(torch.arange(N, device=dev))
```

- [ ] **Step 2: Implement target sampling**

Add to `WarehouseRLEnv`:

```python
    def _sample_targets(self, env_ids: torch.Tensor) -> None:
        """Pick a commanded category per env → goal_id one-hot, target box, and matching zone goal."""
        from env.curriculum import goal_id_onehot
        if env_ids.numel() == 0:
            return
        for e in env_ids.tolist():
            c = int(torch.randint(0, 3, (1,), device=self.device))
            self.box_cat_idx[e] = c
            names = self._boxes_by_cat[self._cat_names[c]]
            self.target_box_name[e] = names[int(torch.randint(0, len(names), (1,)))]
            self.goal_pos[e] = self._zone_pos[c]            # zone order == category order
        self.goal_id_buf[env_ids] = goal_id_onehot(self.box_cat_idx[env_ids], num_cats=3)
        self.holding[env_ids] = False
```

- [ ] **Step 3: Update `_reset_idx` to sample targets + refresh box pos**

Replace `_reset_idx` (`:315-320`):

```python
    def _reset_idx(self, env_ids) -> None:
        """Sample targets+goals, reset scene, randomize box poses, refresh target box pos."""
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._sample_targets(env_ids_t)
        super()._reset_idx(env_ids)
        self._randomize_box_poses(env_ids_t)
        self._refresh_target_box_pos(env_ids_t)
```

- [ ] **Step 4: Implement box-pos refresh + grasp/attach update**

Add to `WarehouseRLEnv`:

```python
    def _refresh_target_box_pos(self, env_ids: torch.Tensor | None = None) -> None:
        """Write each env's commanded box xyz (env-local) into self.box_pos."""
        ids = range(self.num_envs) if env_ids is None else env_ids.tolist()
        for e in ids:
            box = self.scene[self.target_box_name[e]]
            self.box_pos[e] = box.data.root_pos_w[e] - self.scene.env_origins[e]

    def update_grasp(self) -> None:
        """Evaluate grasp-success / drop, set holding + one-shot events, and kinematically carry."""
        from env.grasp import grasp_success, grasp_lost
        self._refresh_target_box_pos()
        robot = self.scene["robot"]
        ee = robot.body_names.index("panda_hand")
        base = robot.body_names.index("base_link")
        self.ee_pos = robot.data.body_pos_w[:, ee] - robot.data.body_pos_w[:, base]
        ee_world = robot.data.body_pos_w[:, ee]
        j = robot.joint_names.index("panda_finger_joint1")
        closed = robot.data.joint_pos[:, j] < 0.0175           # < half-open
        box_lift = torch.stack([
            self.scene[self.target_box_name[e]].data.root_pos_w[e, 2] for e in range(self.num_envs)
        ])
        newly = grasp_success(self.ee_pos, self.box_pos, closed, box_lift) & (~self.holding)
        lost = grasp_lost(self.holding, self.ee_pos, self.box_pos) | (~closed & self.holding)
        self.grasp_event = newly
        self.drop_event = lost & (~self._box_in_any_zone())
        self.holding = (self.holding | newly) & (~lost)
        self._carry_held_boxes(ee_world)

    def _box_in_any_zone(self) -> torch.Tensor:
        """(N,) bool: target box xy within 1.5 m of its commanded zone center (env-local)."""
        return torch.norm(self.box_pos[:, :2] - self.goal_pos[:, :2], dim=-1) < 1.5

    def _carry_held_boxes(self, ee_world: torch.Tensor) -> None:
        """Teleport each held box to the EE (kinematic carry) so physics grip isn't required."""
        for e in range(self.num_envs):
            if not bool(self.holding[e]):
                continue
            box = self.scene[self.target_box_name[e]]
            state = box.data.root_state_w[e:e+1].clone()
            state[:, 0:3] = ee_world[e:e+1]
            state[:, 7:13] = 0.0  # zero velocities
            box.write_root_state_to_sim(state, env_ids=torch.tensor([e], device=self.device))
```

- [ ] **Step 5: Map external (6,) action + call grasp update in the Gym wrapper**

In `WarehouseGymEnv.__init__`, set the v2 spaces (`:348-357`):

```python
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "pixels":   spaces.Box(0.0, 1.0, shape=(3, IMG_HW, IMG_HW), dtype=np.float32),
                "position": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "heading":  spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
                "goal":     spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "goal_id":  spaces.Box(0.0, 1.0, shape=(3,), dtype=np.float32),
                "ee_pos":   spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "gripper":  spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "holding":  spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "box_pos":  spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            }
        )
```

Replace `_unwrap_obs` (`:388-402`):

```python
    def _unwrap_obs(self, obs: dict) -> dict[str, torch.Tensor]:
        policy = obs["policy"]
        if not isinstance(policy, dict):
            raise RuntimeError("ObservationsCfg returned non-dict 'policy'. Set concatenate_terms=False.")
        return {k: policy[k] for k in
                ("pixels", "position", "heading", "goal", "goal_id", "ee_pos", "gripper", "holding", "box_pos")}
```

Replace `step` (`:411-420`):

```python
    def step(self, action):
        """Apply (6,) action [base_lin, base_ang, ee_dx, ee_dy, ee_dz, gripper]."""
        from env.action_pickup import split_action
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(self.device, dtype=torch.float32)
        if action.ndim == 1:
            action = action.unsqueeze(0).expand(self.num_envs, -1)
        action = action.clamp(-1.0, 1.0).to(self.device, dtype=torch.float32)
        base2, ee3, grip1 = split_action(action)
        base3 = self._base_cmd(base2)                       # (N,3) base joint velocities
        internal = torch.cat([base3, ee3, grip1], dim=-1)   # (N,7) base(3)+ik(3)+gripper(1)
        obs, reward, terminated, truncated, info = self._env.step(internal)
        self._env.update_grasp()                            # set holding + grasp/drop events, carry box
        return self._unwrap_obs(obs), reward, terminated, truncated, info
```

Note: `_base_cmd` already accepts a `(N,2)` tensor and returns `(N,3)` — `base2` matches its existing signature unchanged.

- [ ] **Step 6: Switch reward + termination cfgs to the staged set**

Update the `warehouse_reward` import (`:41-48`) to add the new fns:

```python
from env.warehouse_reward import (
    collision_penalty, time_penalty, out_of_bounds,
    approach_box_distance, carry_distance, grasp_success_reward,
    drop_penalty, pickup_delivered, pickup_delivered_reward,
)
```

Replace `RewardsCfg` (`:189-202`) and `TerminationsCfg` (`:206-212`):

```python
@configclass
class RewardsCfg:
    """Staged pick-place reward (see spec §4). Phase switches on env.holding."""
    approach  = RewTerm(func=approach_box_distance,    weight=-0.01)
    grasp     = RewTerm(func=grasp_success_reward,     weight=5.0)
    carry     = RewTerm(func=carry_distance,           weight=-0.01)
    deliver   = RewTerm(func=pickup_delivered_reward,  weight=10.0)
    time_pen  = RewTerm(func=time_penalty,             weight=-0.005)
    collision = RewTerm(func=collision_penalty,        weight=5.0)
    drop      = RewTerm(func=drop_penalty,             weight=-2.0)


@configclass
class TerminationsCfg:
    """Episode end conditions (pickup)."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success  = DoneTerm(func=pickup_delivered)
    bounds   = DoneTerm(func=out_of_bounds, params={"half_extent_x": 9.5, "half_extent_y": 14.5})
```

Bump episode length in `WarehouseEnvCfg.__post_init__` (`:235`):

```python
        self.episode_length_s = 100.0  # 100s x 10Hz = 1000 steps (nav+grasp+carry+place)
```

- [ ] **Step 7: Commit**

```bash
git add env/warehouse_env.py
git commit -m "feat(env): pickup runtime — targets, grasp+attach, 6-dim action map, staged reward"
```

---

## Task 9: Interface-contract integration test (needs GPU + isaaclab)

**Files:**
- Modify: `tests/test_env.py`

Update the contract assertions to v2: action shape `(6,)`, obs has the 9 v2 keys with correct shapes, `goal_id` is one-hot, `holding` is 0/1. Runs only with `conda activate isaaclab` on the GPU box.

- [ ] **Step 1: Update the contract assertions**

In `tests/test_env.py`, set the expected action shape to `(6,)` and the expected obs keys to
`{"pixels","position","heading","goal","goal_id","ee_pos","gripper","holding","box_pos"}`
with shapes `pixels=(N,3,64,64)`, `position/goal/ee_pos/box_pos=(N,3)`, `heading=(N,2)`, `goal_id=(N,3)`, `gripper/holding=(N,1)`. Add:

```python
    # goal_id is a valid one-hot over 3 categories
    gid = obs["goal_id"]
    assert gid.shape[-1] == 3
    assert torch.allclose(gid.sum(dim=-1), torch.ones(gid.shape[0], device=gid.device))
    # holding is 0 or 1
    assert torch.all((obs["holding"] == 0) | (obs["holding"] == 1))
```

- [ ] **Step 2: Run on the GPU box**

Run: `conda activate isaaclab && python tests/test_env.py --num_envs 1`
Expected: ALL PASS, including the new v2 shape + one-hot + holding checks, with the camera ON.

If `panda_hand` is not in `robot.body_names`, print them once and pick the actual Franka hand link:
add a temporary `print(robot.body_names)` in `WarehouseRLEnv._validate` and use the correct EE link name (commonly `panda_hand`), then update `ee_position`, `update_grasp`, and the `arm_ik` `body_name`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_env.py
git commit -m "test(env): v2 interface contract (6-dim action, pickup obs keys)"
```

---

## Task 10: Sync config

**Files:**
- Modify: `configs/env_config.yaml`

Bring the canonical YAML in line with the pickup env. (`CLAUDE.md`/`environment.md` were already updated 2026-06-08.)

- [ ] **Step 1: Update the YAML**

Edit `configs/env_config.yaml` so item count is 18, action is the 6-dim `[base_lin, base_ang, ee_dx, ee_dy, ee_dz, gripper]`, `episode.length_steps: 1000`, the reward block lists `approach(-0.01)/grasp(5.0)/carry(-0.01)/deliver(10.0)/time(-0.005)/collision(5.0)/drop(-2.0)`, and the obs list matches the 9 v2 keys. Keep robot `type: ridgeback_franka`.

- [ ] **Step 2: Verify YAML parses**

Run: `python -c "import yaml; yaml.safe_load(open('configs/env_config.yaml')); print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add configs/env_config.yaml
git commit -m "docs(config): sync env_config.yaml to pickup (18 boxes, 6-dim action, 1000 steps)"
```

---

## Self-Review

**Spec coverage** (spec §2–§7):
- §2 obs v2 (`goal_id` + ee_pos/gripper/holding/box_pos, drop goal_emb) → Tasks 7, 8.
- §3 action `(6,)` + DifferentialIKController + gripper → Tasks 2, 6, 8.
- §4 staged reward (approach/grasp/carry/deliver/time/collision/drop) → Tasks 5, 8.
- §4b CA-SLOPE / Visual HER → **out of scope here** (owners P5/P3, separate plans). This plan delivers the env hooks they consume (`goal_id`, staged dense terms, `box_pos`).
- §5 scene: ~18 reachable boxes, episode 1000 → Tasks 1, 8.
- §7 curriculum: `goal_id` always on, `goal` anneal helper, `box_pos` unannealed → Task 4 (helpers) + Task 8 (wiring).

**Gaps flagged:**
- Goal-xyz anneal is *available* (`anneal_goal`) but not yet driven by a schedule in `goal_position` (Phase-1..3 use alpha=1.0). If Phase-4 anneal is needed before training, add a task: multiply `goal` by `env.goal_alpha` in `goal_position`, decremented by a curriculum callback.
- EE link name `panda_hand` assumed; Task 9 Step 2 has the fallback to verify against `robot.body_names`.
- Kinematic-attach carry teleports the box to the EE each step (robust for RL); a physically-grasped variant is a later refinement, not needed for v1.
- Per-env Python loops in `_sample_targets`/`update_grasp` are fine at num_envs=1 (current VRAM cap); vectorize if num_envs grows.

**Placeholder scan:** none — every code step has full code (config-only YAML edit in Task 10 is described field-by-field).

**Type consistency:** `split_action`→(base2,ee3,grip1) consumed in Task 8 Step 5; `grasp_success`/`grasp_lost` signatures match Task 3; reward fns read the same `env.ee_pos/box_pos/holding/goal_pos/grasp_event/drop_event` set in Task 8; obs fns read `env.goal_id_buf/holding/box_pos` set in Task 8; `_base_cmd((N,2))→(N,3)` unchanged from current code.
