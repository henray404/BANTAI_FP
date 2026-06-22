# recording/recorder.py
# Person 5 — full trajectory recorder + reader. Pure stdlib (no torch/Isaac) → testable on a Mac.
#
# A "run" = one scenario episode. We record TWO artifacts so the run is fully reconstructable:
#   <name>.csv       — one row per control step, EVERY per-step quantity (flat, analyzable).
#   <name>.meta.json — recorded ONCE: scenario + env config + joint name order + final summary.
#
# The per-step schema is DYNAMIC: the recorder writes whatever keys the first row carries (so the
# Isaac extractor can add q_<joint>/qd_<joint> columns for all joints without this file knowing the
# robot). The reader reconstructs types from the header + meta. The contract is: every row has the
# SAME keys, and joint column order is mirrored in meta["joint_names"] for replay.

"""Stream a scenario's full per-step state to CSV + a JSON metadata/summary sidecar."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class TrajectoryRecorder:
    """Write one scenario run: per-step rows to <name>.csv, metadata/summary to <name>.meta.json.

    Usage:
        rec = TrajectoryRecorder("runs/heavy/dreamer_seed0", metadata={...})
        rec.add({"step": 0, "t": 0.0, "a_base_lin": ..., "base_x": ..., "q_panda_joint1": ...})
        ...
        rec.set_summary({"success": 1, "return": 42.0, "steps": 118})
        rec.close()
    """

    def __init__(self, path: str | Path, metadata: dict[str, Any] | None = None):
        """Open a run at `path` (without extension). Metadata is held until close()/first add()."""
        p = Path(path)
        if p.suffix == ".csv":
            p = p.with_suffix("")
        p.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path = p.with_suffix(".csv")
        self.meta_path = p.with_name(p.name + ".meta.json")
        self._metadata: dict[str, Any] = dict(metadata or {})
        self._summary: dict[str, Any] = {}
        self._fieldnames: list[str] | None = None
        self._f = None
        self._writer: csv.DictWriter | None = None
        self._n_rows = 0

    def add(self, row: dict[str, Any]) -> None:
        """Append one per-step row. The first call fixes the column order from the row's keys."""
        if self._writer is None:
            self._fieldnames = list(row.keys())
            self._f = open(self.csv_path, "w", newline="")
            self._writer = csv.DictWriter(self._f, fieldnames=self._fieldnames)
            self._writer.writeheader()
            self._metadata["columns"] = self._fieldnames
            self._flush_meta()
        if list(row.keys()) != self._fieldnames:
            missing = set(self._fieldnames) ^ set(row.keys())
            raise ValueError(f"recorder row keys changed mid-run (diff: {sorted(missing)}). "
                             "Every step must record the same fields.")
        self._writer.writerow(row)
        self._n_rows += 1

    def set_summary(self, summary: dict[str, Any]) -> None:
        """Record the run-level outcome (success, return, steps, grasp/deliver step) for ranking."""
        self._summary.update(summary)

    def _flush_meta(self) -> None:
        """Write/overwrite the JSON sidecar (metadata + current summary + row count)."""
        payload = {**self._metadata, "summary": self._summary, "n_steps": self._n_rows}
        with open(self.meta_path, "w") as mf:
            json.dump(payload, mf, indent=2, default=_json_safe)

    def close(self) -> None:
        """Flush the final metadata+summary and close the CSV."""
        self._flush_meta()
        if self._f is not None:
            self._f.close()
            self._f = None

    def __enter__(self):
        """Context-manager entry."""
        return self

    def __exit__(self, *exc):
        """Context-manager exit always closes the file."""
        self.close()


class TrajectoryReader:
    """Load a recorded run: typed per-step rows + the metadata sidecar (for replay/analysis)."""

    def __init__(self, path: str | Path):
        """Open a run at `path` (.csv, .meta.json, or the stem)."""
        p = Path(path)
        stem = p.with_suffix("") if p.suffix in (".csv", ".json") else p
        if stem.name.endswith(".meta"):
            stem = stem.with_name(stem.name[: -len(".meta")])
        self.csv_path = stem.with_suffix(".csv")
        self.meta_path = stem.with_name(stem.name + ".meta.json")

        with open(self.meta_path) as mf:
            self.meta: dict[str, Any] = json.load(mf)
        self.joint_names: list[str] = self.meta.get("joint_names", [])
        self.summary: dict[str, Any] = self.meta.get("summary", {})

        self.rows: list[dict[str, Any]] = []
        with open(self.csv_path, newline="") as cf:
            for raw in csv.DictReader(cf):
                self.rows.append({k: _parse(v) for k, v in raw.items()})

    def joint_pos(self, row: dict[str, Any]) -> list[float]:
        """Reconstruct the joint-position vector for a row, ordered like meta['joint_names']."""
        return [float(row[f"q_{n}"]) for n in self.joint_names]

    def joint_vel(self, row: dict[str, Any]) -> list[float]:
        """Reconstruct the joint-velocity vector (zeros if not recorded)."""
        return [float(row.get(f"qd_{n}", 0.0)) for n in self.joint_names]

    def __len__(self) -> int:
        """Number of recorded steps."""
        return len(self.rows)


def _parse(v: str):
    """CSV cell → int/float when possible, else the raw string (empty → None)."""
    if v == "" or v is None:
        return None
    try:
        i = int(v)
        return i
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def _json_safe(o):
    """Fallback JSON encoder: tolist() for arrays/tensors, else str()."""
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)
