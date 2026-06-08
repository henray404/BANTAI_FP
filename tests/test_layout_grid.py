# Pure-python layout math tests (no Isaac Sim needed). Run: pytest tests/test_layout_grid.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.layout_grid import island_rack_positions, item_specs


def test_island_rack_count():
    racks = island_rack_positions((-6.0, 0.0, 6.0), (8.0, 1.0, -5.0), 1.5)
    assert len(racks) == 18  # 9 islands x 2 racks


def test_island_rack_offsets():
    racks = island_rack_positions((0.0,), (0.0,), 1.5)
    assert racks == [(-0.75, 0.0, 0.0), (0.75, 0.0, 0.0)]


def test_item_specs_count_and_cycle():
    racks = island_rack_positions((-6.0, 0.0, 6.0), (8.0, 1.0, -5.0), 1.5)
    items = item_specs(racks, (0.21, 0.32, 0.52), (2.0, 6.0, 12.0), 1.5)
    assert len(items) == 18
    assert items[0][0] == "fragile_0"
    assert items[0][2] == 2.0  # mass
    assert items[1][0] == "regular_0"
    assert items[1][2] == 6.0
    assert items[2][0] == "heavy_0"
    assert items[2][2] == 12.0
    assert items[3][0] == "fragile_1"


def test_item_z_sits_on_shelf():
    items = item_specs([(0.0, 0.0, 0.0)], (0.21, 0.32, 0.52), (2.0, 6.0, 12.0), 1.5)
    name, size, mass, pos = items[0]
    assert size == 0.21
    assert mass == 2.0
    assert abs(pos[2] - (1.5 + 0.105)) < 1e-9


def test_target_box_specs_count_and_reachability():
    """18 target boxes, one per rack, resting on the bottom shelf within Franka reach."""
    from env.layout_grid import TARGET_BOX_SPECS, RACK_POSITIONS, BOTTOM_SHELF_Z

    assert len(TARGET_BOX_SPECS) == len(RACK_POSITIONS) == 18
    # box xy must equal its rack center (sitting on that rack's shelf)
    rack_xy = {(round(rx, 4), round(ry, 4)) for rx, ry, _ in RACK_POSITIONS}
    for name, size, mass, pos in TARGET_BOX_SPECS:
        # box rests on the bottom shelf deck: center z == shelf surface + half the cube
        assert abs(pos[2] - (BOTTOM_SHELF_Z + size / 2.0)) < 1e-6, \
            f"{name} not on bottom shelf (z={pos[2]})"
        # box top must stay within the Franka vertical reach ceiling (~1.35m)
        assert pos[2] + size / 2.0 <= 1.35, f"{name} top {pos[2] + size/2.0:.2f}m out of reach"
        assert (round(pos[0], 4), round(pos[1], 4)) in rack_xy, f"{name} not on a rack ({pos[0]},{pos[1]})"


def test_target_box_categories_cycle():
    """Categories cycle fragile/regular/heavy across the 18 racks."""
    from env.layout_grid import TARGET_BOX_SPECS
    cats = [name.split("_")[0] for name, *_ in TARGET_BOX_SPECS]
    assert cats.count("fragile") == cats.count("regular") == cats.count("heavy") == 6
