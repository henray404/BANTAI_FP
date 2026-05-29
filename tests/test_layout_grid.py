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
    items = item_specs(racks, (0.21, 0.32, 0.52), 1.5)
    assert len(items) == 18
    assert items[0][0] == "fragile_0"
    assert items[1][0] == "regular_0"
    assert items[2][0] == "heavy_0"
    assert items[3][0] == "fragile_1"


def test_item_z_sits_on_shelf():
    items = item_specs([(0.0, 0.0, 0.0)], (0.21, 0.32, 0.52), 1.5)
    name, size, pos = items[0]
    assert size == 0.21
    assert abs(pos[2] - (1.5 + 0.105)) < 1e-9
