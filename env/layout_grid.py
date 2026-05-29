# layout_grid.py
# Pure-python warehouse layout math. NO isaaclab import -> unit-testable standalone.
"""Grid generators for rack-island positions and item placement."""

from __future__ import annotations

_CATEGORIES = ("fragile", "regular", "heavy")


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
    shelf_z: float,
) -> list[tuple[str, float, tuple[float, float, float]]]:
    """One box per rack, cycling category by index (fragile/regular/heavy).

    sizes = (fragile_m, regular_m, heavy_m) edge lengths in meters.
    Returns list of (name, size, (x, y, shelf_z + size/2)).
    """
    counters = {c: 0 for c in _CATEGORIES}
    out: list[tuple[str, float, tuple[float, float, float]]] = []
    for i, (x, y, _z) in enumerate(rack_positions):
        cat = _CATEGORIES[i % 3]
        size = sizes[i % 3]
        name = f"{cat}_{counters[cat]}"
        counters[cat] += 1
        out.append((name, size, (x, y, shelf_z + size / 2.0)))
    return out
