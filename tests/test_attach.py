from env.attach import GRASP_JOINT_NAME, grasp_joint_path


def test_grasp_joint_path():
    p = grasp_joint_path("/World/envs/env_0/box_fragile_0")
    assert p == f"/World/envs/env_0/box_fragile_0/{GRASP_JOINT_NAME}"


def test_grasp_joint_path_strips_trailing_slash():
    assert grasp_joint_path("/World/envs/env_0/box/") == f"/World/envs/env_0/box/{GRASP_JOINT_NAME}"
