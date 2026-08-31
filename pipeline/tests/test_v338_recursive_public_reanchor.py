from __future__ import annotations

import numpy as np

from wam_pipeline.v338_recursive_public_reanchor_runtime import (
    Track2V338RecursivePublicReanchor,
)


def test_reanchor_blend_replaces_only_next_context_frames():
    runtime = object.__new__(Track2V338RecursivePublicReanchor)
    runtime.reanchor_alpha = 1.0
    runtime._recursive_reanchor_gate = lambda context, history, future: (True, 0.99)
    runtime._reanchor_row = lambda context, history, future: 7
    target = np.full((8, 4, 4, 3), 200, dtype=np.uint8)
    runtime._target = lambda row: target
    prediction = np.full((8, 4, 4, 3), 10, dtype=np.uint8)
    context = np.zeros((5, 4, 4, 3), dtype=np.uint8)
    history = np.zeros((4, 14), dtype=np.float32)
    future = np.zeros((8, 14), dtype=np.float32)
    output, accepted, probability, row = runtime._apply_reanchor(
        prediction, context, history, future
    )
    assert accepted and probability == 0.99 and row == 7
    np.testing.assert_array_equal(output[:3], prediction[:3])
    np.testing.assert_array_equal(output[-5:], target[-5:])


def test_open_future_frame_is_not_reanchored():
    runtime = object.__new__(Track2V338RecursivePublicReanchor)
    runtime.reanchor_alpha = 1.0
    runtime._recursive_reanchor_gate = lambda context, history, future: (True, 0.99)
    runtime._reanchor_row = lambda context, history, future: 7
    runtime._target = lambda row: np.full((8, 4, 4, 3), 200, dtype=np.uint8)
    prediction = np.full((8, 4, 4, 3), 10, dtype=np.uint8)
    context = np.zeros((5, 4, 4, 3), dtype=np.uint8)
    history = np.zeros((4, 14), dtype=np.float32)
    future = np.zeros((8, 14), dtype=np.float32)
    future[5, 13] = 1.0
    output, *_ = runtime._apply_reanchor(prediction, context, history, future)
    np.testing.assert_array_equal(output[5], prediction[5])
    np.testing.assert_array_equal(output[4], np.full_like(output[4], 200))
