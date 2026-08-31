import numpy as np

from wam_pipeline.irasim_physical_actions import (
    _active_arm,
    _matrix_to_euler_zyx,
    _quat_wxyz_to_matrix,
    _relative_ee_actions,
)


def test_identity_quaternion_and_rotation_round_trip():
    rotation = _quat_wxyz_to_matrix(np.asarray([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(rotation, np.eye(3), atol=1e-7)
    np.testing.assert_allclose(_matrix_to_euler_zyx(rotation), np.zeros(3), atol=1e-7)


def test_relative_actions_use_local_translation_and_bridge_scaling():
    endpose = np.asarray(
        [[0, 0, 0, 1, 0, 0, 0], [0.1, -0.2, 0.05, 1, 0, 0, 0]], dtype=np.float64
    )
    result = _relative_ee_actions(endpose, np.asarray([1.0, 0.0]), arm_identity=-1.0)
    np.testing.assert_allclose(result[0, :3], [2.0, -4.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(result[0, 3:6], 0.0, atol=1e-6)
    np.testing.assert_allclose(result[0, 6:], [0.0, -1.0], atol=1e-6)


def test_active_arm_is_selected_episode_wide():
    values = np.zeros((4, 14), dtype=np.float64)
    values[:, 8] = [0.0, 0.2, 0.4, 0.6]
    assert _active_arm(values) == "right"
