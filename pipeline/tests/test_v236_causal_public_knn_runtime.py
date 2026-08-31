"""Unit tests for the v236 causal public-retrieval blend."""

import numpy as np

from wam_pipeline.v236_causal_public_knn_runtime import (
    Track2V236CausalPublicKNNBlend,
)


def test_causal_blend_only_changes_closed_right_gripper_frames():
    parent = np.zeros((8, 256, 256, 3), dtype=np.uint8)
    retrieval = np.full_like(parent, 100)
    actions = np.zeros((8, 14), dtype=np.float32)
    actions[:, 13] = 1.0
    actions[[2, 3, 6], 13] = 0.0

    actual = Track2V236CausalPublicKNNBlend.causal_blend(
        parent, retrieval, actions
    )

    assert np.all(actual[[0, 1, 4, 5, 7]] == 0)
    assert np.all(actual[[2, 3, 6]] == 70)


def test_causal_blend_validates_action_shape():
    frames = np.zeros((8, 256, 256, 3), dtype=np.uint8)
    try:
        Track2V236CausalPublicKNNBlend.causal_blend(
            frames, frames, np.zeros((7, 14), dtype=np.float32)
        )
    except ValueError as exc:
        assert "[8,14]" in str(exc)
    else:
        raise AssertionError("invalid future action shape was accepted")
