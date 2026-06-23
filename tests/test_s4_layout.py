# tests/test_s4_layout.py — pure-python tests for the s4 transfer-scene layout helpers.
# No Isaac Lab import: exercises env/layout_grid scale_shelf_levels + rest_box_specs only.

"""s4 transfer env: shelf scaling + box re-resting (smaller rack, box on 2nd shelf level)."""

from __future__ import annotations

import math

from env.layout_grid import (
    BASE_RACK_SCALE,
    extra_shelf_box_specs,
    rest_box_specs,
    scale_shelf_levels,
)

_BASE_LEVELS = (0.72351, 1.32528, 1.92566)  # warehouse_scene._BASE_SHELF_LEVELS


def test_scale_levels_identity_at_base_scale():
    """At the reference rack scale the shelf levels are unchanged."""
    assert scale_shelf_levels(_BASE_LEVELS, BASE_RACK_SCALE) == _BASE_LEVELS


def test_scale_levels_shrinks_linearly():
    """Shrinking the rack lowers every shelf by the same factor (0.007/0.01 = 0.7)."""
    out = scale_shelf_levels(_BASE_LEVELS, 0.007)
    for z_old, z_new in zip(_BASE_LEVELS, out):
        assert math.isclose(z_new, z_old * 0.7, rel_tol=1e-9)
    # mid shelf drops from ~1.325 m to ~0.928 m — into Franka reach.
    assert math.isclose(out[1], 0.928, abs_tol=1e-3)


def _specs():
    # (name, size, mass, (x, y, z)); z is ignored by rest_box_specs (recomputed).
    return [
        ("fragile_0", 0.21, 2.0, (-6.75, 8.0, 99.0)),
        ("regular_0", 0.32, 6.0, (-5.25, 8.0, 99.0)),
        ("heavy_0", 0.52, 12.0, (-0.75, 8.0, 99.0)),
    ]


def test_rest_default_is_bottom_shelf():
    """No lift + unscaled levels → every box rests on the bottom shelf (level 0)."""
    out = rest_box_specs(_specs(), _BASE_LEVELS)
    for (_n, size, _m, (_x, _y, z)) in out:
        assert math.isclose(z, _BASE_LEVELS[0] + size / 2.0, rel_tol=1e-9)


def test_rest_preserves_xy_and_identity():
    """Re-resting changes only z — name/size/mass/x/y are untouched."""
    src = _specs()
    out = rest_box_specs(src, _BASE_LEVELS)
    for (n0, s0, m0, (x0, y0, _z0)), (n1, s1, m1, (x1, y1, _z1)) in zip(src, out):
        assert (n0, s0, m0, x0, y0) == (n1, s1, m1, x1, y1)


def test_extra_box_added_on_mid_shelf():
    """extra_shelf_box_specs ADDS a box per rack on the mid shelf (bottom box untouched)."""
    levels = scale_shelf_levels(_BASE_LEVELS, 0.007)
    base = rest_box_specs(_specs(), levels)          # 3 bottom boxes on the shrunk rack
    extra = extra_shelf_box_specs(_specs(), levels, rack_indices=[0], level=1)
    assert len(extra) == 1
    name, size, mass, (x, y, z) = extra[0]
    # same rack 0 → same category prefix, x/y, size/mass
    assert name == "fragile_shelf1_0" and name.startswith("fragile")
    assert (size, mass, x, y) == (0.21, 2.0, -6.75, 8.0)
    # rests on the mid shelf, center ~0.928 + 0.105
    assert math.isclose(z, levels[1] + 0.21 / 2.0, rel_tol=1e-9)
    # combined set: 3 bottom + 1 mid, unique names, mid box is a NEW box (not a moved one)
    combined = base + extra
    assert len(combined) == 4
    assert len({n for n, *_ in combined}) == 4


def test_extra_box_racks_all():
    """rack_indices for every rack → one extra mid box per rack."""
    levels = scale_shelf_levels(_BASE_LEVELS, 0.007)
    extra = extra_shelf_box_specs(_specs(), levels, rack_indices=[0, 1, 2], level=1)
    assert [n for n, *_ in extra] == ["fragile_shelf1_0", "regular_shelf1_1", "heavy_shelf1_2"]
