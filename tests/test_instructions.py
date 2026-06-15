# tests/test_instructions.py — pure-CPU unit tests (no Isaac, no CLIP).
#   pytest tests/test_instructions.py -v
"""Unit tests for perception.language.instructions zone↔category mapping."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perception.language.instructions import (
    ALL_INSTRUCTIONS,
    INSTRUCTION_BY_CATEGORY,
    INSTRUCTION_BY_ZONE,
    ZONE_INSTRUCTIONS,
)


def test_three_zones_in_order():
    zones = [z for z, _, _ in ZONE_INSTRUCTIONS]
    assert zones == ["zone_A", "zone_B", "zone_C"]  # MUST match ZONE_SPECS order


def test_category_mapping():
    assert INSTRUCTION_BY_CATEGORY["fragile"] == "deliver small box to orange zone"
    assert INSTRUCTION_BY_CATEGORY["heavy"] == "deliver large box to purple zone"


def test_all_instructions_unique_and_complete():
    assert len(ALL_INSTRUCTIONS) == 3
    assert len(set(ALL_INSTRUCTIONS)) == 3
    assert set(INSTRUCTION_BY_ZONE) == {"zone_A", "zone_B", "zone_C"}
