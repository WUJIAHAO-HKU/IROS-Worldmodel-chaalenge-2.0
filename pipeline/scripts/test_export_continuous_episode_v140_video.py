import numpy as np

from export_continuous_episode_v140_video import (
    chunked_open_loop_indices,
    hardest_horizon_indices,
    verify_timeline,
)


def test_chunked_timeline_covers_every_global_frame_once():
    starts = np.arange(125)
    entries = chunked_open_loop_indices(starts, 8)
    global_times = [int(starts[window]) + 5 + horizon for window, horizon, _ in entries]
    assert global_times == list(range(5, 137))
    assert len(entries) == 132
    assert all(0 <= horizon < 8 for _, horizon, _ in entries)


def test_hardest_horizon_is_continuous_and_fixed():
    starts = np.arange(125)
    entries = hardest_horizon_indices(starts, 8)
    assert [horizon for _, horizon, _ in entries] == [7] * 125
    assert [int(starts[window]) + 5 + horizon for window, horizon, _ in entries] == list(range(12, 137))


def test_verify_timeline_accepts_one_frame_shift():
    timeline = np.arange(15, dtype=np.uint8)[:, None, None, None]
    target = np.stack([timeline[index + 5:index + 13] for index in range(3)])
    names = np.asarray([f"episode7_{index:05d}.npz" for index in range(3)])
    assert np.array_equal(verify_timeline(names, target), np.arange(3))
