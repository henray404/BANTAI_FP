# experiments/scenarios.py
# Person 5 — scenario definitions for the headless eval harness.
#
# Coordinates mirror the real scene constants so the toy traces line up with the Isaac env:
#   - delivery zones:  env/warehouse_scene.py ZONE_SPECS (y = -12, x = -6 / 0 / 6)
#   - rack islands:    env/layout_grid.py ISLAND_COLS_X/ROWS_Y, bottom shelf z = 0.72351
#   - receiving spawn: CONTEXT.md (north, x:-8..+8, y:+11..+14)
# Category index order matches the goal_id one-hot: 0 fragile/orange, 1 regular/cyan, 2 heavy/purple.

"""Eval scenarios: one per category, with spawn / target-box / delivery-zone coordinates."""

from __future__ import annotations

from dataclasses import dataclass, field

BOTTOM_SHELF_Z = 0.72351  # env/layout_grid.BOTTOM_SHELF_Z


@dataclass(frozen=True)
class Scenario:
    """One eval episode setup. All xyz are env-local metres (same frame as the obs dict)."""

    name: str
    category: int                       # 0 fragile, 1 regular, 2 heavy
    color: str
    spawn_xy: tuple[float, float]       # base start (receiving area, north)
    box_xyz: tuple[float, float, float] # target box resting pose (on a bottom shelf)
    goal_xyz: tuple[float, float, float]  # delivery zone centre (matching color)

    @property
    def goal_id(self) -> tuple[float, float, float]:
        """One-hot category vector, same layout as env/curriculum.goal_id_onehot."""
        oh = [0.0, 0.0, 0.0]
        oh[self.category] = 1.0
        return tuple(oh)


# Default scenario set: pick a representative box per category at the middle island row (y=+1),
# carry south to the matching color zone (y=-12). Spawn at receiving north (0, +12).
DEFAULT_SCENARIOS: list[Scenario] = [
    Scenario(
        name="fragile_orange",
        category=0,
        color="orange",
        spawn_xy=(0.0, 12.0),
        box_xyz=(-6.0, 1.0, BOTTOM_SHELF_Z),
        goal_xyz=(-6.0, -12.0, 0.01),
    ),
    Scenario(
        name="regular_cyan",
        category=1,
        color="cyan",
        spawn_xy=(0.0, 12.0),
        box_xyz=(0.0, 1.0, BOTTOM_SHELF_Z),
        goal_xyz=(0.0, -12.0, 0.01),
    ),
    Scenario(
        name="heavy_purple",
        category=2,
        color="purple",
        spawn_xy=(0.0, 12.0),
        box_xyz=(6.0, 1.0, BOTTOM_SHELF_Z),
        goal_xyz=(6.0, -12.0, 0.01),
    ),
]
