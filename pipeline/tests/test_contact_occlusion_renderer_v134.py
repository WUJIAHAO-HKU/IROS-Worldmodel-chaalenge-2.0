import numpy as np

from evaluate_contact_occlusion_head_v13 import (
    _observed_dark_material,
    _remove_stale_gripper,
)


def test_material_support_excludes_white_outline_and_caps_texture():
    source = np.full((40, 40, 3), 180, np.uint8)
    source[12:28, 12:28] = 28
    source[11, 11:29] = 245
    semantic = np.zeros((40, 40), bool)
    semantic[11:29, 11:29] = True
    support, fallback = _observed_dark_material(source, semantic, 92.0)
    assert support[12:28, 12:28].all()
    assert not support[11, 11:29].any()
    assert float(fallback.max()) <= 92.0


def test_rejected_old_gripper_is_inpainted_from_green_bottle():
    query = np.zeros((64, 64, 3), np.uint8)
    query[:] = (30, 160, 35)
    query[26:38, 10:24] = 24
    current_gripper = np.zeros((64, 64), bool)
    current_gripper[26:38, 34:48] = True
    current_bottle = np.ones((64, 64), bool)
    probability = np.zeros((3, 64, 64), np.float32)
    probability[1] = 0.95
    probability[2, 26:38, 34:48] = 0.99
    cleaned, stale = _remove_stale_gripper(
        query.astype(np.float32), current_gripper, current_bottle,
        probability, 0.08, 1, 3.0
    )
    assert stale[30, 15]
    assert cleaned[30, 15, 1] > 100
    assert cleaned[30, 15].mean() > 55
