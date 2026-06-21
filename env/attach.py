# attach.py
# Person 4 — physics grasp: runtime fixed-joint attach/detach between the Franka hand and a box.
#
# Replaces the kinematic teleport carry (WarehouseRLEnv._carry_held_boxes). On grasp, a
# UsdPhysics.FixedJoint welds the box rigid body to panda_hand at the relative transform present
# at creation time, so PhysX holds the box under physics (collisions + weight on the arm) instead
# of snapping it to the EE each step. On release the joint prim is removed.
#
# Mirrors env.warehouse_scene._weld_robot_world_links (same FixedJoint pattern, already proven in
# this repo for the base weld). USD-only; no Isaac Lab managers, so it runs from update_grasp().
#
# VERIFY ON FIRST SIM RUN (scripts/tune_arm.py prints the resolved prim paths):
#   * panda_hand prim path resolves (find_descendant_path returns non-None).
#   * After attach, the box tracks the EE under physics and does NOT fall.
#   * Runtime joint add/remove is honored by the GPU PhysX pipeline (num_envs=1). If the joint is
#     ignored (box falls) or errors, flip WarehouseRLEnv CARRY_MODE back to "kinematic".

"""USD fixed-joint attach/detach for physics-based box grasping."""

from __future__ import annotations

GRASP_JOINT_NAME = "grasp_joint"


def grasp_joint_path(box_prim_path: str) -> str:
    """Stage path of the fixed joint authored under a box prim (one per box, pure string op)."""
    return f"{box_prim_path.rstrip('/')}/{GRASP_JOINT_NAME}"


def find_descendant_path(stage, root_path: str, name: str) -> str | None:
    """Return the stage path of the first descendant of `root_path` whose prim name == `name`.

    Used to resolve the panda_hand LINK prim inside the Ridgeback-Franka articulation USD without
    assuming its nesting depth (the camera mount proves links sit under Robot/, but depth varies).
    """
    from pxr import Usd

    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return None
    for prim in Usd.PrimRange(root):
        if prim.GetName() == name:
            return prim.GetPath().pathString
    return None


def attach_box(stage, hand_prim_path: str, box_prim_path: str) -> bool:
    """Weld `box_prim_path` to `hand_prim_path` with a FixedJoint. Idempotent. Returns created?.

    PhysX freezes the body0→body1 relative transform at definition time, so the box stays wherever
    it is relative to the hand at grasp. Call after the box has been positioned at the EE.
    """
    from pxr import Sdf, UsdPhysics

    jp = Sdf.Path(grasp_joint_path(box_prim_path))
    if stage.GetPrimAtPath(jp).IsValid():
        return False  # already attached
    joint = UsdPhysics.FixedJoint.Define(stage, jp)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(hand_prim_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(box_prim_path)])
    return True


def detach_box(stage, box_prim_path: str) -> bool:
    """Remove the FixedJoint under a box prim, if present. Idempotent. Returns removed?."""
    jp = grasp_joint_path(box_prim_path)
    if not stage.GetPrimAtPath(jp).IsValid():
        return False
    stage.RemovePrim(jp)
    return True
