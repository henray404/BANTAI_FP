# Warehouse Layout v2 — Design Spec

**Date**: 2026-05-30
**Author**: Person 1 (Environment & Integration)
**Status**: Approved (brainstorm) -> pending spec review
**Scope**: VISUAL/SPATIAL layout redesign only. Mission unchanged (navigate-to-colored-zone). Reward/curriculum plan (Pass 3-5) untouched.

---

## 1. Goal

Make the env look like a real warehouse (IKEA / NVIDIA reference images) using owned NVIDIA DT assets, while staying trainable on RTX 5050 8GB. Task stays: robot spawns in receiving (north), navigates to a randomly-assigned colored delivery zone (south), avoiding rack islands. Success = within radius of goal zone.

Decisions locked in brainstorm:
- Look-only (mission unchanged)
- Room: rectangular 20x30m
- Structure: island blocks (open navigation, not corridor aisles)
- Density: 9 islands (3x3 grid), ~18 racks
- Props: boxes-on-pallets + floor markings (cones/signs) + 1 forklift + empty pallets
- Walls: primitive concrete cuboids (building shell deferred - SubUSDs 1.2GB, bug-log risk)
- Spawn: receiving-north
- num_envs: start at 2 (VRAM safety), bump to 4 only if headless test passes

---

## 2. Room Geometry

| Param | Value | Notes |
|---|---|---|
| ROOM_HALF_X | 10.0 | room width 20m (x in [-10,+10]) |
| ROOM_HALF_Y | 15.0 | room depth 30m (y in [-15,+15]) |
| WALL_H | 6.0 | unchanged |
| WALL_T | 0.3 | unchanged |
| env_spacing | **32.0** | must exceed largest dim 30m + buffer (was 22.0) |

4 walls: N(y=+15), S(y=-15), E(x=+10), W(x=-10). Wall lengths adjust to rectangular footprint.

Ground: concrete physics material (static 0.8 / dynamic 0.6, already in Pass 1).

---

## 3. Top-Down Layout

```
                NORTH WALL  y=+15
 ============ RECEIVING (y +11..+14) ============
   [forklift]      [pallet-asm] [pallet-asm]
        robot spawn: x[-8,+8], y[+11,+14], yaw random
 ------------------------------------------------
   I1            I2            I3          y=+8
  [r][r]        [r][r]        [r][r]
   I4            I5            I6          y=+1
  [r][r]        [r][r]        [r][r]
   I7            I8            I9          y=-5
  [r][r]        [r][r]        [r][r]
   x=-6          x=0           x=+6
       [empty-pallet]   [cone] [cone]
 ------------------------------------------------
 ============ SHIPPING (y -9..-14) ==============
   [ZONE_A]       [ZONE_B]       [ZONE_C]   y=-12
    orange         cyan           purple
   (-6,-12)       (0,-12)        (+6,-12)
       [warn-sign]          [warn-sign]
                SOUTH WALL  y=-15
```

---

## 4. Storage: 9 Island Blocks (3x3)

Island centers:
- X columns: {-6, 0, +6}
- Y rows: {+8, +1, -5}

Each island = 2 racks. Placement within island: 2 racks offset along X by `ISLAND_RACK_DX` (default 1.5m, i.e. rack at cx +-0.75).
> **VERIFY**: Rack_A01 X/Y footprint unknown (explore_rack only measured Z). Extend explore_rack to print X/Y bbox; tune ISLAND_RACK_DX so racks don't overlap.

Inter-island aisle: column spacing 6m - island width ~3m ~= 3m open lane. Row spacing 6-7m. Open island-block feel.

Racks: `assets/Shelving/Racks/Rack_A/Rack_A01_PR_NVD_01.usd`, scale (0.01,0.01,0.01), static AssetBaseCfg + collision.

### Items (visual landmarks, NOT goals)
18 boxes, one per rack, cycling category by rack index:
- idx%3==0 -> fragile (CubeBox_A03_21cm)
- idx%3==1 -> regular (CubeBox_A05_32cm)
- idx%3==2 -> heavy (CubeBox_A07_52cm)

Position: rack (x,y) + z = RACK_SHELF_Z(1.5) + size/2. Static USD, scale (0.01,...), collision. Provides CLIP/YOLO visual data for P3/P4.

---

## 5. Delivery Zones (GOAL - unchanged)

3 colored 3x3m slabs at south:

| Zone | Color | Pos |
|---|---|---|
| zone_A | orange (1.0,0.9,0.0) | (-6, -12, 0.01) |
| zone_B | cyan (0.0,0.9,0.9) | (0, -12, 0.01) |
| zone_C | purple (0.7,0.0,0.9) | (+6, -12, 0.01) |

ZONE_SPECS unchanged in structure (goal sampling reads these).

---

## 6. Props (low counts - VRAM)

All USD authored in cm -> scale (0.01,0.01,0.01), static AssetBaseCfg. Copy from `C:\Users\Henry\Downloads\Warehouse_NVD@10013\Assets\DigitalTwin\Assets\Warehouse\` into project `assets/`.

| Prop | Count | USD | Placement |
|---|---|---|---|
| Forklift | 1 | Equipment/Forklifts/Forklift_A/Forklift_A01_PR_V_NVD_01.usd | (-5, +12) receiving, static obstacle |
| Boxes-on-pallet | 2 | Shipping/Cardboard_Boxes_on_Pallet/Pallet_Asm_A/Pallet_Asm_A01_66x66x46cm + A04_120x122x75cm | (3,+12),(5,+12) receiving staging |
| Empty pallet | 2 | Shipping/Pallets/Plastic/Economy_A/EconomyPlasticPallet_A01_PR_NVD_01.usd | (-3,-7),(4,-7) |
| Traffic cone | 4 | Safety/Cones/Heavy-Duty_Traffic/HeavyDutyTrafficCone_A02_71cm_PR_V_NVD_01.usd | island/aisle corners |
| Warning sign | 2 | Safety/Floor_Signs/Warning_A/WarningSign_A01_PR_NVD_01.usd | (-6,-9.5),(6,-9.5) near zones |

Props with collision_props = static obstacles (forklift, pallets) contribute to collision reward (Pass 3). Cones/signs cosmetic (collision optional).

---

## 7. Mission Flow (unchanged task)

1. Robot spawns receiving-north: pose_range x[-8,8], y[+11,+14], yaw[-pi,pi]
2. Goal = random colored zone (A/B/C) at south
3. Robot navigates through/around 3x3 island field
4. Success: within reached_goal threshold of goal zone -> +1, episode ends
5. Failure: timeout 60s, or out_of_bounds

Now requires real traversal across storage field (richer than v1 open-field).

---

## 8. Training Param Changes

| Param | v1 | v2 | Reason |
|---|---|---|---|
| Room | 20x20 | 20x30 | rectangular warehouse |
| env_spacing | 22.0 | **32.0** | > 30m largest dim |
| episode_length_s | 45 | **60** | north->south ~25m @1m/s + navigation |
| num_envs | 4 | **2** | more geometry/textures on 8GB; bump to 4 if test OK |
| spawn pose_range | +-3m central | x[-8,8] y[+11,14] | receiving-north start |
| out_of_bounds | half_extent 9.5 (square) | **half_extent_x=9.5, half_extent_y=14.5** | rectangular room - termination fix |

`out_of_bounds` signature change: add separate x/y half-extents (required for rectangular room; not a reward-tuning change).

---

## 9. Code Changes

**warehouse_scene.py**:
- ROOM_HALF -> ROOM_HALF_X=10, ROOM_HALF_Y=15; wall factory uses both
- RACK_POSITIONS -> 9-island generator (3x3, 2 racks each = 18)
- ITEM_SPECS -> 18 boxes cycling category
- New prop USD path constants + `_prop_cfg` factory (static USD, scale 0.01, optional collision)
- Add forklift/pallet-asm/empty-pallet/cone/sign cfgs to WarehouseSceneCfg
- ISLAND_RACK_DX constant (verify footprint)

**warehouse_env.py**:
- env_spacing 22->32, episode 45->60, num_envs default 2
- EventCfg spawn pose_range -> receiving-north
- (out_of_bounds bounds passed via TerminationsCfg params)

**warehouse_reward.py**:
- out_of_bounds: half_extent -> half_extent_x, half_extent_y

**configs/env_config.yaml**: mirror all above.

**asset_sandbox/scripts/explore_rack.py**: also print X/Y bbox (footprint for ISLAND_RACK_DX).

**Copy assets**: forklift, pallet-asm x2, empty pallet, cone, warning sign -> project assets/ (self-contained check; flag MDL-external like Rack).

---

## 10. VRAM Budget (RTX 5050 8GB)

Per env: 18 racks + 18 boxes + 4 walls + 3 zones + 1 forklift + 2 pallet-asm + 2 pallets + 4 cones + 2 signs ~= 54 prims. x2 envs.
Geometry instanced (unique mesh loaded once); cost mainly unique textures (forklift, pallet-asm have PBR). Conservative num_envs=2; headless OOM test before training. If pass -> try 4.

---

## 11. Test / Verify Plan (Pass 2 extended)

1. Copy assets, confirm each USD loads standalone (no missing MDL fatal)
2. explore_rack.py -> rack X/Y footprint -> set ISLAND_RACK_DX
3. explore_scene.py -> visual check: islands spaced, boxes on shelves, props placed, no overlap/floating
4. test_env.py -> obs contract still holds (pixels/position/goal/goal_emb shapes)
5. Headless num_envs=2 -> confirm no OOM; check VRAM; try 4
6. Confirm collision geometry on racks/forklift/pallets (gate for Pass 3 collision reward)

---

## 12. Out of Scope

- Building shell (Warehouse_A modular) - heavy SubUSDs, defer
- Functional zone mission (receiving->storage->shipping flow) - mission stays navigate-to-zone
- Reward rebalance / collision reward / curriculum - Pass 3-5, separate
- Pickup mechanic, custom AMR, domain randomization
