# layout_grid.py
# Pure-python warehouse layout math. NO isaaclab import -> unit-testable standalone.
"""Grid generators for rack-island positions and item placement."""

from __future__ import annotations

_CATEGORIES = ("fragile", "regular", "heavy")

# ── Module-level constants matching warehouse_scene.py defaults ───────────────
# Exported so tests can verify box specs without importing Isaac Lab / pxr.
BOX_SMALL_SIZE: float = 0.21
BOX_MED_SIZE:   float = 0.32
BOX_LARGE_SIZE: float = 0.52
BOX_MASSES: tuple[float, float, float] = (2.0, 6.0, 12.0)

ISLAND_COLS_X:    tuple[float, ...] = (-6.0, 0.0, 6.0)
ISLAND_ROWS_Y:    tuple[float, ...] = (8.0, 1.0, -5.0)
ISLAND_RACK_DX:   float = 1.5
BOX_FRONT_OFFSET: float = 0.5  # meters in -y from rack center


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
    front_offset: float,
) -> list[tuple[str, float, float, tuple[float, float, float]]]:
    """18 floor-level target boxes, one per rack, within Franka reach.

    Category cycles fragile/regular/heavy by rack index (6 of each).
    Box center z = size/2 (floor-resting). Box placed front_offset meters
    in -y from rack center (toward shipping area).
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
        out.append((name, size, mass, (rx, ry - front_offset, size / 2.0)))
    return out


# ── Pre-computed module-level instances (importable without Isaac Lab) ────────
RACK_POSITIONS: list[tuple[float, float, float]] = island_rack_positions(
    ISLAND_COLS_X, ISLAND_ROWS_Y, ISLAND_RACK_DX
)

TARGET_BOX_SPECS: list[tuple[str, float, float, tuple[float, float, float]]] = target_box_specs(
    RACK_POSITIONS,
    (BOX_SMALL_SIZE, BOX_MED_SIZE, BOX_LARGE_SIZE),
    BOX_MASSES,
    BOX_FRONT_OFFSET,
)
