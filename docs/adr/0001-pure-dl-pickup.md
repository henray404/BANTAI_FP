# ADR 0001 — Pure-DL Pickup: Drop CLIP and YOLO

- **Status:** Accepted
- **Date:** 2026-06-08
- **Spec:** `docs/superpowers/specs/2026-06-08-pure-dl-pickup-redesign.md`

## Context

Original framing: "Text-Conditioned World Model for Visual Category-Aware Warehouse Robot" — Deep Learning + NLP (CLIP) + PCD (YOLO), 5 people. Box category was to be perceived: a text instruction → CLIP 512-dim embedding for the goal, and YOLO object detection to identify the box. This pulled two heavy perception stacks into a project whose graded core is the DreamerV3 world model, multiplying integration risk on an 8GB RTX 5050.

## Decision

Pure Deep Learning. Remove CLIP and YOLO entirely.

- No text instructions, no `goal_emb` (512-dim language embedding removed).
- No object detection. Box identity/category is **given** directly in the observation as `goal_id` — a 3-dim one-hot over `[orange, cyan, purple]` — not perceived.
- `goal_id` selects both the target box and the matching delivery zone.
- DreamerV3 visual world model remains the DL core. Franka arm becomes active for a full pick → carry → place task.

New title: **"Visual Goal-Conditioned World Model for Warehouse Pickup"**.

## Consequences

- P2 removes the 512→64 CLIP projection; `goal_id` (3-dim) feeds the RSSM directly.
- P3 (was YOLO) → Policy + Visual HER. P4 (was CLIP) → Manipulation.
- Research novelty retained without perception: **CA-SLOPE** reads category from `goal_id`, not a detector (see ADR-0003 / spec §4b).
- Lower VRAM and integration surface; perception of category is explicitly **out of scope** for v1.
- Supersedes category/text framing in `CLAUDE.md`, `docs/project_overview.md`, `docs/environment.md`, `docs/timeline_terbaru.md`.
