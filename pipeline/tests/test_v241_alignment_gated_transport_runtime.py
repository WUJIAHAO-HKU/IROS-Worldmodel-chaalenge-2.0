from __future__ import annotations

import numpy as np

from wam_pipeline.v241_alignment_gated_transport_runtime import (
    Track2V241AlignmentGatedTransport,
    _path_length,
    _transport_descriptor,
)


def test_post_grasp_and_blend_gate() -> None:
    history = np.ones((4, 14), dtype=np.float32)
    future = np.ones((8, 14), dtype=np.float32)
    assert not Track2V241AlignmentGatedTransport._post_grasp(history, future)
    history[-1, 13] = 0.0
    future[:, 13] = 0.0
    assert Track2V241AlignmentGatedTransport._post_grasp(history, future)
    future[0, 13] = 1.0
    parent = np.zeros((8, 256, 256, 3), dtype=np.uint8)
    target = np.full_like(parent, 100)
    blended = Track2V241AlignmentGatedTransport._blend_with_alpha(
        parent, target, future, 0.5
    )
    assert np.all(blended[0] == 0)
    assert np.all(blended[1:] == 50)


def test_alignment_gate_rejects_opposite_motion() -> None:
    runtime = Track2V241AlignmentGatedTransport.__new__(
        Track2V241AlignmentGatedTransport
    )
    runtime.motion_scale = 1.0
    runtime.library_history = np.zeros((1, 4, 14), dtype=np.float32)
    runtime.library_future = np.zeros((1, 8, 14), dtype=np.float32)
    runtime.library_future[0, :, 7] = np.linspace(0.1, 1.0, 8)
    history = np.zeros((4, 14), dtype=np.float32)
    future = np.zeros((8, 14), dtype=np.float32)
    future[:, 7] = np.linspace(0.1, 1.0, 8)
    aligned = runtime._transport_alpha(history, future, 0, 0.0)
    future[:, 7] *= -1
    opposite = runtime._transport_alpha(history, future, 0, 0.0)
    assert np.isclose(aligned, 1.0)
    assert opposite == 0.0
    assert _transport_descriptor(history[None], future[None]).shape == (1, 48)
    assert _path_length(history[None], future[None])[0] > 0
