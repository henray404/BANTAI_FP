# tests/test_recording.py
# Person 5 — recorder/reader round-trip. Pure stdlib, runs on a Mac (no torch/Isaac).
# Validates the plumbing that the Isaac state extractor feeds: dynamic joint columns, metadata
# sidecar, summary, and faithful joint-vector reconstruction for replay.

import json

from recording.recorder import TrajectoryRecorder, TrajectoryReader

JOINTS = ["dummy_base_prismatic_x", "dummy_base_revolute_z", "panda_joint1", "panda_finger_joint1"]


def _row(step):
    """A synthetic full row shaped like state_extractor.step_row (core fields + q_/qd_ per joint)."""
    r = {
        "step": step, "t": round(0.1 * step, 4),
        "a_base_lin": 1.0, "a_base_ang": 0.0, "a_ee_dx": 0.0, "a_ee_dy": 0.0, "a_ee_dz": 0.0, "a_grip": 1.0,
        "base_x": 0.5 * step, "base_y": 1.0, "base_z": 0.1,
        "base_qw": 1.0, "base_qx": 0.0, "base_qy": 0.0, "base_qz": 0.0,
        "base_roll_deg": 0.0, "base_pitch_deg": 0.0, "base_yaw_deg": 90.0,
        "ee_x": 0.0, "ee_y": 0.0, "ee_z": 0.6, "ee_qw": 1.0, "ee_qx": 0.0, "ee_qy": 0.0, "ee_qz": 0.0,
        "ee_base_x": 0.4, "ee_base_y": 0.0, "ee_base_z": 0.6,
        "box_x": 6.0, "box_y": 1.0, "box_z": 0.72,
        "box_qw": 1.0, "box_qx": 0.0, "box_qy": 0.0, "box_qz": 0.0,
        "gripper": 1.0, "holding": 0, "grasp_event": 0, "drop_event": 0,
        "goal_x": 6.0, "goal_y": -12.0, "goal_z": 0.0,
        "reward": -0.13, "slope_reward": "", "terminated": 0, "truncated": 0, "contact_force_n": 0.0,
    }
    for i, n in enumerate(JOINTS):
        r[f"q_{n}"] = round(0.01 * step + 0.1 * i, 6)
    for n in JOINTS:
        r[f"qd_{n}"] = 0.0
    return r


def test_record_read_roundtrip(tmp_path):
    meta = {"run_id": "t", "seed": 0, "category": "heavy", "joint_names": JOINTS, "control_dt": 0.1}
    rec = TrajectoryRecorder(tmp_path / "run", metadata=meta)
    for s in range(5):
        rec.add(_row(s))
    rec.set_summary({"success": 1, "return": 42.0, "steps": 5})
    rec.close()

    reader = TrajectoryReader(tmp_path / "run")
    assert len(reader) == 5
    assert reader.joint_names == JOINTS
    assert reader.summary["success"] == 1
    # faithful joint reconstruction for replay (step 3, joint 2 -> 0.01*3 + 0.1*2 = 0.23)
    q = reader.joint_pos(reader.rows[3])
    assert len(q) == len(JOINTS)
    assert abs(q[2] - 0.23) < 1e-9
    # numeric cells parsed back to numbers, not strings
    assert isinstance(reader.rows[0]["base_x"], float)
    assert isinstance(reader.rows[0]["step"], int)


def test_meta_sidecar_has_scenario(tmp_path):
    meta = {"run_id": "t", "seed": 7, "category": "fragile", "goal_zone_xyz": [-6.0, -12.0, 0.01],
            "joint_names": JOINTS, "target_box_name": "box_3"}
    rec = TrajectoryRecorder(tmp_path / "run2", metadata=meta)
    rec.add(_row(0))
    rec.close()
    payload = json.loads((tmp_path / "run2.meta.json").read_text())
    assert payload["seed"] == 7
    assert payload["target_box_name"] == "box_3"
    assert payload["columns"][0] == "step"          # column order captured
    assert payload["n_steps"] == 1


def test_row_key_mismatch_raises(tmp_path):
    rec = TrajectoryRecorder(tmp_path / "run3", metadata={"joint_names": JOINTS})
    rec.add(_row(0))
    bad = _row(1)
    del bad["box_x"]
    try:
        rec.add(bad)
        assert False, "expected ValueError on changed row keys"
    except ValueError:
        pass
    finally:
        rec.close()
