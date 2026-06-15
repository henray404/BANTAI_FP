# perception/language/instructions.py
# Person 4 — text instruction templates per delivery zone.
#
# Zone↔category mapping mirrors env/warehouse_scene.ZONE_ITEM_MAP + ZONE_SPECS.
# Order MUST match env.warehouse_scene.ZONE_SPECS (zone_A, zone_B, zone_C) so a
# zone index from goal_pos lines up with the right instruction/embedding.

"""Per-zone language instructions for CLIP text encoding."""

from __future__ import annotations

# index → (zone_name, category, instruction). Order == ZONE_SPECS order.
ZONE_INSTRUCTIONS: list[tuple[str, str, str]] = [
    ("zone_A", "fragile", "deliver small box to orange zone"),
    ("zone_B", "regular", "deliver medium box to cyan zone"),
    ("zone_C", "heavy",   "deliver large box to purple zone"),
]

INSTRUCTION_BY_ZONE: dict[str, str] = {z: t for z, _, t in ZONE_INSTRUCTIONS}
INSTRUCTION_BY_CATEGORY: dict[str, str] = {c: t for _, c, t in ZONE_INSTRUCTIONS}
ALL_INSTRUCTIONS: list[str] = [t for _, _, t in ZONE_INSTRUCTIONS]
