# Bug: Hand-rolled arm IK holds the EE target in the ROOT (world-origin) frame → arm drags/flails when the base drives

**Date:** 2026-06-22
**File:** scripts/drive_env_v2.py (also latent in scripts/drive_robot.py)
**Status:** [x] Fixed (code)  [ ] Open  [ ] Workaround — UNVERIFIED in sim (Blackwell; user to confirm)

---

## Error Message
```
(no exception — visual misbehavior)
User report: "ini pas aku coba ternyata armnya banyak tingkah"
("the arm misbehaves / flails a lot") in scripts/drive_env_v2.py teleop.
```

## What I Was Doing
Built `scripts/drive_env_v2.py` — a lightweight 1-rack/1-box teleop for tuning the
Ridgeback-Franka grasp/gripper (see "New tooling" below). Borrowed the absolute-pose
DifferentialIKController loop from `scripts/drive_robot.py`. When driving the base toward the
box with the arm active, the arm flailed / dragged backward.

First hypotheses (partially right, not the root cause):
- `--ee_sens` too high → fast target slew. (A contributor only.)
- Base-acceleration coupling / DLS near singularity. (Minor.)
- Gripper jamming the oversized box. (Real, but separate — see Notes.)

## Root Cause
The hand-rolled IK expressed the EE hold target in the **articulation ROOT frame**, which for the
welded Ridgeback-Franka is the `world` link **pinned at the world origin** — NOT the moving chassis.

- `_ee_pose_base()` computes the EE pose relative to `robot.data.root_pose_w`.
- The Ridgeback base is floating in the shipped USD, so it is welded at spawn
  (`_weld_robot_world_links`, see `2026-06-09_ridgeback-floating-base.md`). After the weld the
  articulation root = the `world` link, fixed at the spawn origin; `base_link` moves via the dummy
  prismatic/revolute joints. This is the exact IsaacLab #1268 trap the **observation** code already
  avoids (`robot_position`/`robot_heading` read `body_pos_w["base_link"]`, not `root_pos_w`).
- The IK code did NOT avoid it. The EE target captured at spawn is therefore a **world-fixed point
  near the origin**. As the base drives away, holding that world-fixed target forces the arm to
  reach back toward the origin → it saturates fully extended → "flail/drag". The further you drive,
  the worse it gets.

It is **not** the 7-DOF redundancy and **not** the Jacobian indexing: the arm-column slice
(`jac_joint_ids`, with the +6 floating-base offset) and `ee_jacobi_idx` match Isaac Lab's canonical
`scripts/tutorials/05_controllers/run_diff_ik.py` convention exactly.

**Training is NOT affected by this bug** — the training arm action uses Isaac Lab's official
`DifferentialInverseKinematicsActionCfg` term, which handles the body/base offset internally. (The
training arm has a *different* problem — relative-mode IK ratchet/drift — which is why it is
currently frozen; see `2026-06-21_frozen-arm-snags-racks.md`. That is tracked separately for the
future "Lane B / active-arm" work.)

## Fix
Hold the EE target **relative to `base_link`** (the moving chassis) and re-express it into the root
frame each step, so the target rides with the robot.

```python
# capture (in main): EE hold target RELATIVE TO base_link, not root
base_pos_w  = robot.data.body_pos_w[:, base_link_idx]
base_quat_w = robot.data.body_quat_w[:, base_link_idx]
ee_pos_w    = robot.data.body_pos_w[:, ee_body_idx]
ee_quat_w   = robot.data.body_quat_w[:, ee_body_idx]
ee_target, ee_quat_des = subtract_frame_transforms(base_pos_w, base_quat_w, ee_pos_w, ee_quat_w)

# each step (in _arm_ik_targets): base-relative target -> world -> root frame
tgt_pos_w, tgt_quat_w = combine_frame_transforms(base_pos_w, base_quat_w, ee_target_b, ee_quat_b)
root_w = robot.data.root_pose_w
tgt_pos_r, tgt_quat_r = subtract_frame_transforms(root_w[:, 0:3], root_w[:, 3:7], tgt_pos_w, tgt_quat_w)
arm_ik.set_command(torch.cat([tgt_pos_r, tgt_quat_r], dim=-1))
return arm_ik.compute(ee_pos_r, ee_quat_r, jacobian, joint_pos)  # ee_pos_r/quat_r still root-frame
```

Also added two flail-tamers (helpful, not the root fix):
- `--arm_smooth` (default 0.2): EXP low-pass on the IK joint targets — kills high-freq twitch from
  fast key slew + base-acceleration transients. `1.0` = raw IK (no smoothing).
- `--ee_sens` default lowered 0.003 → 0.002 (calmer reach).

`python -m py_compile scripts/drive_env_v2.py` → clean. Sim run still pending (Blackwell hardware).

## New tooling (logged here per "catat semuanya")
`scripts/drive_env_v2.py` — lightweight grasp/gripper-tuning teleop, created 2026-06-22.
- **Minimal scene:** robot + 1 box + 1 rack only. No onboard camera, no shelf decks, no props,
  walls, or zones. Reuses `RIDGEBACK_FRANKA_CFG`, `_rack_cfg`, `_item_cfg` from `warehouse_scene`.
- **Why:** `drive_env.py` runs the full task env (≈150 prims + camera) and renders ~3 fps windowed
  on the 8 GB RTX 5050 — render-bound, not physics. v2 strips it to tune the grasp without the lag.
- **FPS levers baked in:** camera OFF (biggest render save), `--render_every 4` (~50 Hz render cap),
  ~3 prims vs ~150. Free extras for the user: shrink the Isaac window, check `nvidia-smi` for VRAM.
- **Grasp telemetry (~5 Hz):** prints ee↔box **surface** distance, gripper open/closed, and whether
  the env's shipped `grasp_success()` would FIRE — the exact signal to tune `GRIP_RADIUS_M` / the
  arm home pose against. Uses the same `env.grasp.grasp_success` the real env uses (no duplicate
  threshold logic).
- Flags: `--box_size {0.21,0.32,0.52}`, `--box_dist`, `--grip_radius`, `--no_rack`,
  `--render_every`, `--arm_smooth`, `--ee_sens`, `--lin/--ang/--strafe`.

## Notes
- **Gripper jams the oversized box (separate, expected in v2).** v2 is a bare scene with NO magnetic
  weld (unlike the real env). Boxes are 21–52 cm but the gripper opening is only 3.5 cm, so closing
  K against the box just collides → contact jolt. For tuning, do NOT close on the box — read the
  telemetry. (The real env welds the box on a proximity grab, so it never jams there.)
- **C3 reminder (from the 2026-06-22 audit / `docs/project/training_readiness_2026-06-22.md`).** With
  the current tucked home pose the frozen hand sits ~1 m behind+up the base, so `surface_dist` may
  never drop below `GRIP_RADIUS_M`. Apply the "forward-down try" `home_joint_pos` angles in
  `configs/env_config.yaml` and re-test in v2.
- **Next (Lane B).** Once `--arm_smooth`/`--ee_sens` + home pose feel right in v2, port the control
  to training: switch the arm action from relative-mode to absolute-hold + clamp, set
  `arm_active=True`, and carry the smoothing — gives the world model clean, predictable arm dynamics.
- Related: `2026-06-09_ridgeback-floating-base.md` (the weld), `2026-06-16_base-drive-doublecounts-spawn-yaw.md`
  (#1268 base-frame trap, base side), `2026-06-21_frozen-arm-snags-racks.md` (why the training arm is frozen).

---

## Follow-on: workspace calibration + clamps (2026-06-22)

After the frame fix, the arm still misbehaved in Z ("ngawur") and strained at the clamp box edges.
Added clamps + a data-driven calibration pass.

**Clamps added to `drive_env_v2.py` (all should port to training / Lane B):**
- Soft joint-limit clamp on the IK output (`--joint_margin`, default 0.95 of the hard USD range) —
  the `DifferentialIKController` does not clamp its own output, so an unreachable target drove
  joints past their limit and oscillated.
- Asymmetric box clamp (`--reach_xy` 0.3, `--reach_z` 0.18) — Z hits the elbow/wrist limits fastest.
- Target leash (`--lead` 0.08 m) — the target may lead the actual EE by at most `lead`, so it never
  outruns reach and strains at an unreachable point ("maksa di limit").
- Radial workspace clamp (`--reach_r`/`--reach_rmin`) — couples x/y/z (the dexterous set is a shell,
  not a box). Proper fix for "Z fine at some X, ngawur at others".
- EE-target EXP low-pass (`--arm_smooth` 0.2) — smooths twitch from fast keys + base-accel transients.

**Calibration tooling:** `--calib <csv>` mode frees all clamps (sweep the envelope) and logs per step:
ee xyz (base frame), radius from the shoulder (panda_link0), `min_joint_margin`, `pose_err`, and
**manipulability** = sqrt(det(Jp·Jpᵀ)) (singularity sensor). Center is computed live at runtime, so
only r_min/r_max need fitting.

**Calibration result (from `calib/arm_envelope.csv`, ~8.3k steps):**
- Reachable EE: x∈[-0.50,1.12], y∈[-0.84,0.51], z∈[0.356,1.41]; radius∈[0.62,1.14]. Arm DOES descend
  to z≈0.36 (the earlier "z stuck at 1.21" was just insufficient teleop input, not a limit).
- Manipulability vs radius: healthy band radius 0.65–0.90 (manip 0.067–0.087, peak ~0.75–0.80);
  0.90–0.95 ok (manip 0.048, = the spawn region, radius 0.934); radius >0.95 = sparse leash-off
  overshoot (unstable) → exclude. `pose_err` is contaminated by the leash being off, so it was NOT
  used as an envelope signal.
- **Fitted radial clamp: r_max = 0.95, r_min = 0.50** → now the DEFAULTS in `drive_env_v2.py`
  (`--reach_r 0.95 --reach_rmin 0.50`), radial clamp ON by default.
- `min_joint_margin` grazes 0 across the whole workspace → one wrist joint is pinned at its limit by
  the FIXED top-down orientation constraint. Handled by the soft 0.95 clamp; relaxing orientation
  (use the 7-DOF redundancy) is future work.
- The tucked SPAWN pose was the low-manipulability spot (manip ≈ 0.012) → that is the "Z ngawur at
  rest". Radial clamp does not move the rest config out of it, so the home pose was retuned:

**Home pose retuned (C3 fix), `configs/env_config.yaml` `home_joint_pos`:** tucked → forward-down:
`joint2 -0.569→0.3`, `joint4 -2.810→-2.0`, `joint6 2.0→2.3`. Moves the rest EE lower/forward (toward
the floor box) and out of the low-manip tucked region. **UNVERIFIED in sim** — affects TRAINING spawn
+ frozen-arm grasp geometry too; verify with `drive_env_v2.py` (no `--calib`) before trusting.

**Better calibration method (noted for next time):** the manual teleop sweep is slow and incomplete.
The kinematics is deterministic — no learned model is needed. The efficient approach is a headless
**reachability/capability map**: sample joint configs within limits → FK → record EE + manipulability
+ margin → auto-build the envelope (thousands of samples in seconds, no rendering). A small optimizer
(e.g. CMA-ES / scipy) can then pick the home pose that maximizes manipulability at the grasp-down
config subject to reaching the box. TODO: `scripts/calibrate_arm.py`.
