# Bug/Change Log: USD Assets + Size-Based Item Categories

**Date:** 2026-05-22
**Author:** Person 1

## Summary

Two intentional overrides from original CLAUDE.md rules, confirmed with user:

### 1. Policy Override: External USD Assets Allowed

- **Original rule:** `❌ Never use external USD assets — primitive shapes only`
- **Override:** User explicitly approved using NVIDIA Warehouse Digital Twin asset pack
- **Asset pack:** `C:\Users\Henry\Downloads\Warehouse_NVD@10013`
- **Pack source:** NVIDIA Omniverse Digital Twin Warehouse (NVD @10013)
- **Assets copied to workspace `assets/`:**
  - `assets/Shelving/Racks/Rack_A/Rack_A01_PR_NVD_01.usd` (self-contained USDC, 2.1 MB)
  - `assets/Shipping/Cardboard_Boxes/Cube_A/` (all sizes + materials/, 118 MB)
- **Assets referenced by config path (NOT copied due to size):**
  - Building walls: replaced with primitive cuboids (Warehouse_A SubUSDs = 1.2 GB, too heavy)
- **VRAM impact:** 4 envs x (1 rack type + 3 box types) acceptable for RTX 5050 8 GB

### 2. Convention Change: Color to Size for Item Categories

- **Original convention:** colored boxes (red=fragile, green=regular, blue=heavy)
- **New convention:** cardboard boxes, size encodes category
  - CubeBox_A03_21cm = fragile (small)
  - CubeBox_A05_32cm = regular (medium)
  - CubeBox_A07_52cm = heavy (large)
- **Rationale:** Cardboard USD assets have realistic textures; color override would look
  unrealistic. Size variation is visually distinct for CLIP (Person 4).
- **Team impact:** Person 4 (CLIP) should note visual category signal is now SIZE not COLOR.
  Delivery zones (zone_A/B/C) still use distinct colors for goal identification.

## Layout Changes (20 x 20 m room)

Previous layout: 8 m env_spacing, 6 m half-extent
New layout:
- env_spacing: 22.0 m (must exceed room size)
- Room: 20 x 20 m (ROOM_HALF = 10.0)
- 6 racks at y = +6.0, x = [-7.5, -4.5, -1.5, +1.5, +4.5, +7.5]
- 3 delivery zones at y = -7.0, x = [-6.0, 0.0, +6.0]
- Walls: primitive cuboids (light concrete grey, 6 m tall)

## RACK_SHELF_Z Note

RACK_SHELF_Z = 2.0 is an estimate. Rack_A01 USD actual height unknown until first
simulation run. If boxes float or clip through racks, tune this constant in
warehouse_scene.py and rerun.
