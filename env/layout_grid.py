# layout_grid.py
# Pure-python warehouse layout math. NO isaaclab import -> unit-testable standalone.
"""Grid generators for rack-island positions and item placement."""

from __future__ import annotations

import math

_CATEGORIES = ("fragile", "regular", "heavy")


def avoidance_heading(
    target_xy: tuple[float, float],
    base_xy: tuple[float, float],
    islands: list[tuple[float, float]],
    influence: float,
    push: float,
    skip_near_target: float,
) -> tuple[float, float]:
    """Potential-field steering: (desired_heading_rad, distance_to_target).

    Unit attraction vector toward `target_xy` plus, for each island within `influence` metres of
    `base_xy`, a repulsion pointing away from that island scaled by closeness. The island within
    `skip_near_target` of the target is ignored (else the base can never reach a box on a shelf).
    Pure geometry — no Isaac. Used by scripts/demo_pickup.py for rack avoidance.
    """
    ax, ay = target_xy[0] - base_xy[0], target_xy[1] - base_xy[1]
    dist = math.hypot(ax, ay)
    sx, sy = ax / (dist or 1e-6), ay / (dist or 1e-6)
    for ix, iy in islands:
        if math.hypot(target_xy[0] - ix, target_xy[1] - iy) < skip_near_target:
            continue  # target sits on this island — don't push away from it
        dx, dy = base_xy[0] - ix, base_xy[1] - iy
        dd = math.hypot(dx, dy)
        if 1e-3 < dd < influence:
            # Inverse-distance (classic potential field): blows up as the base nears the rack,
            # decays to 0 at the influence edge → strong shove when close, open aisles when far.
            w = push * (1.0 / dd - 1.0 / influence)
            sx += w * dx / dd
            sy += w * dy / dd
    return math.atan2(sy, sx), dist

# ── Module-level constants matching warehouse_scene.py defaults ───────────────
# Exported so tests can verify box specs without importing Isaac Lab / pxr.
BOX_SMALL_SIZE: float = 0.21
BOX_MED_SIZE:   float = 0.32
BOX_LARGE_SIZE: float = 0.52
BOX_MASSES: tuple[float, float, float] = (2.0, 6.0, 12.0)

ISLAND_COLS_X:    tuple[float, ...] = (-6.0, 0.0, 6.0)
ISLAND_ROWS_Y:    tuple[float, ...] = (8.0, 1.0, -5.0)
ISLAND_RACK_DX:   float = 1.5
# Bottom shelf surface z (mirrors warehouse_scene.RACK_SHELF_LEVELS[0]). The Franka, mounted
# ~0.5m up on the Ridgeback, reaches ~0 to ~1.35m, so only the BOTTOM shelf (0.72m) is a
# reliable grasp target — mid (1.32m) is borderline, top (1.93m) out of reach.
BOTTOM_SHELF_Z:   float = 0.72351


def island_rack_positions(
    cols_x: tuple[float, ...],
    rows_y: tuple[float, ...],
    rack_dx: float,
) -> list[tuple[float, float, float]]:
    """Return (x, y, z=0) for 2 racks per island over a cols_x x rows_y grid.

    Each island center (cx, cy) yields racks at (cx - rack_dx/2, cy) and
    (cx + rack_dx/2, cy). Row-major order: all islands in rows_y[0] first.
    """
    out: list[tuple[float, float, float]] = []
    for cy in rows_y:
        for cx in cols_x:
            out.append((cx - rack_dx / 2.0, cy, 0.0))
            out.append((cx + rack_dx / 2.0, cy, 0.0))
    return out


def item_specs(
    rack_positions: list[tuple[float, float, float]],
    sizes: tuple[float, float, float],
    masses: tuple[float, float, float],
    shelf_z: float,
) -> list[tuple[str, float, float, tuple[float, float, float]]]:
    """One box per rack, cycling category by index (fragile/regular/heavy).

    sizes = (fragile_m, regular_m, heavy_m) edge lengths in meters.
    masses = (fragile_kg, regular_kg, heavy_kg) weights in kg.
    Returns list of (name, size, mass, (x, y, shelf_z + size/2)).
    """
    counters = {c: 0 for c in _CATEGORIES}
    out: list[tuple[str, float, float, tuple[float, float, float]]] = []
    for i, (x, y, _z) in enumerate(rack_positions):
        cat = _CATEGORIES[i % 3]
        size = sizes[i % 3]
        mass = masses[i % 3]
        name = f"{cat}_{counters[cat]}"
        counters[cat] += 1
        out.append((name, size, mass, (x, y, shelf_z + size / 2.0)))
    return out


def target_box_specs(
    rack_positions: list[tuple[float, float, float]],
    sizes: tuple[float, float, float],
    masses: tuple[float, float, float],
    shelf_z: float,
) -> list[tuple[str, float, float, tuple[float, float, float]]]:
    """18 target boxes, one per rack, resting on the BOTTOM shelf (within Franka reach).

    Category cycles fragile/regular/heavy by rack index (6 of each).
    Box sits on the shelf deck at the rack center: center z = shelf_z + size/2.
    Returns list of (name, size, mass, (x, y, z)).
    """
    counters = {c: 0 for c in _CATEGORIES}
    out: list[tuple[str, float, float, tuple[float, float, float]]] = []
    for i, (rx, ry, _) in enumerate(rack_positions):
        cat  = _CATEGORIES[i % 3]
        size = sizes[i % 3]
        mass = masses[i % 3]
        name = f"{cat}_{counters[cat]}"
        counters[cat] += 1
        out.append((name, size, mass, (rx, ry, shelf_z + size / 2.0)))
    return out


# ── s4 transfer test: scaled rack + box on a higher shelf level ───────────────
# The s4 generalization env (configs/env_config_s4.yaml) physically shrinks the rack and rests one
# target box on the 2nd shelf LEVEL (mid) instead of the bottom. Shrinking the rack lowers every
# shelf surface, dropping the mid shelf into Franka reach — so a model trained only on bottom-shelf
# boxes is tested on a NEW shelf height it never saw. Pure math (no Isaac) so it is unit-testable.

# Reference rack scale the measured shelf levels were captured at (warehouse_scene.RACK_USD scale).
BASE_RACK_SCALE: float = 0.01


def scale_shelf_levels(
    levels: tuple[float, ...], scale: float, base_scale: float = BASE_RACK_SCALE
) -> tuple[float, ...]:
    """Shelf surface z's at a different rack USD scale.

    The rack origin sits on the floor (z=0) so a uniform USD scale scales every shelf height
    linearly: z_new = z_old * (scale / base_scale). scale == base_scale → unchanged.
    """
    f = scale / base_scale
    return tuple(z * f for z in levels)


def rest_box_specs(
    specs: list[tuple[str, float, float, tuple[float, float, float]]],
    shelf_levels: tuple[float, ...],
    level: int = 0,
) -> list[tuple[str, float, float, tuple[float, float, float]]]:
    """Re-rest each box on a shelf `level`, keeping its (x, y).

    Used to drop every existing box onto the (possibly scaled) BOTTOM shelf when the rack shrinks,
    so the bottom boxes still sit on the shrunk deck. Resting z = shelf surface + size/2. With
    unscaled levels and level 0 this reproduces the original bottom-shelf specs exactly.
    """
    out: list[tuple[str, float, float, tuple[float, float, float]]] = []
    for name, size, mass, (x, y, _z) in specs:
        z = shelf_levels[level] + size / 2.0
        out.append((name, size, mass, (x, y, z)))
    return out


def extra_shelf_box_specs(
    base_specs: list[tuple[str, float, float, tuple[float, float, float]]],
    shelf_levels: tuple[float, ...],
    rack_indices: list[int],
    level: int = 1,
) -> list[tuple[str, float, float, tuple[float, float, float]]]:
    """One ADDITIONAL box per rack in `rack_indices`, resting on shelf `level` (default 1 = mid).

    For the s4 transfer test: each chosen rack keeps its bottom-shelf box AND gets a second box on a
    higher shelf. The new box reuses the rack's (x, y) + size/mass/category, with a unique name that
    keeps the category prefix (e.g. "fragile_shelf1_0") so the env still commands it as a target and
    randomizes its x,y on reset, exactly like the bottom-shelf boxes. Resting z = surface + size/2.
    """
    out: list[tuple[str, float, float, tuple[float, float, float]]] = []
    for i in rack_indices:
        name0, size, mass, (x, y, _z) = base_specs[i]
        cat = name0.split("_")[0]
        z = shelf_levels[level] + size / 2.0
        out.append((f"{cat}_shelf{level}_{i}", size, mass, (x, y, z)))
    return out


def shuffled_box_layout(
    specs: list[tuple[str, float, float, tuple[float, float, float]]],
    slots: list[tuple[float, float, int]],
    shelf_levels: tuple[float, ...],
    seed: int | None = None,
) -> list[tuple[str, float, float, tuple[float, float, float]]]:
    """Randomly assign each box to a (rack, shelf-level) slot — the s4 placement randomizer.

    Box SIZE is baked at spawn (Isaac can't cheaply resize a rigid body), so instead of changing
    sizes we permute WHICH box sits in WHICH slot. A box keeps its name/size/mass/category and only
    moves to a random slot; its z is recomputed from the slot's shelf level + its own size so it
    rests correctly. Result: any shelf can hold any size (e.g. big on the mid shelf, small on the
    bottom). `slots` = [(x, y, shelf_level_index), ...], one per box. seed=None → nondeterministic
    (different layout each launch); pass an int to reproduce a layout. len(slots) must == len(specs).
    """
    import random

    if len(slots) != len(specs):
        raise ValueError(f"need one slot per box: {len(slots)} slots vs {len(specs)} boxes")
    order = list(range(len(slots)))
    random.Random(seed).shuffle(order)
    out: list[tuple[str, float, float, tuple[float, float, float]]] = []
    for (name, size, mass, _pos), slot_i in zip(specs, order):
        x, y, level = slots[slot_i]
        out.append((name, size, mass, (x, y, shelf_levels[level] + size / 2.0)))
    return out


# ── Pre-computed module-level instances (importable without Isaac Lab) ────────
RACK_POSITIONS: list[tuple[float, float, float]] = island_rack_positions(
    ISLAND_COLS_X, ISLAND_ROWS_Y, ISLAND_RACK_DX
)

TARGET_BOX_SPECS: list[tuple[str, float, float, tuple[float, float, float]]] = target_box_specs(
    RACK_POSITIONS,
    (BOX_SMALL_SIZE, BOX_MED_SIZE, BOX_LARGE_SIZE),
    BOX_MASSES,
    BOTTOM_SHELF_Z,
)
