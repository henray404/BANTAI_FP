# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

This repo is **single-context**: one `CONTEXT.md` + `docs/adr/` at the repo root.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root.
- **`docs/adr/`** — read ADRs that touch the area you're about to work in.

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront. The producer skill (`/grill-with-docs`) creates them lazily when terms or decisions actually get resolved.

## File structure

```
/
├── CONTEXT.md                         ← domain glossary (not created yet)
├── CLAUDE.md                          ← project context, stack, obs/action contract
├── docs/
│   ├── adr/                           ← architecture decisions (not created yet)
│   │   ├── 0001-pure-dl-pickup.md     ← e.g. drop CLIP/YOLO, goal_id one-hot
│   │   └── 0002-ridgeback-franka.md   ← e.g. replace Carter/Jetbot
│   ├── environment.md                 ← env design doc
│   ├── CHANGES.md                     ← env change log
│   └── superpowers/specs/             ← approved specs
├── env/                               ← warehouse_env / scene / reward / layout_grid
├── scripts/                           ← run_env, drive_robot, smoke_test
└── tests/                             ← test_env, test_layout_grid
```

> ADR filenames above are illustrative — `docs/adr/` does not exist yet. `/grill-with-docs` creates real ADRs lazily as decisions get resolved.

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal — either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/grill-with-docs`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007 — but worth reopening because…_
