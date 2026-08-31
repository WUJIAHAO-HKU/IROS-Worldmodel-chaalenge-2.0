from __future__ import annotations

import json

import numpy as np

from wam_pipeline.v15_runtime import Track2V15Runtime


def make_window(path, value: int, arm: int) -> None:
    context = np.full((5, 256, 256, 3), value, dtype=np.uint8)
    history = np.zeros((4, 14), dtype=np.float32)
    future = np.zeros((8, 14), dtype=np.float32)
    column = arm * 7
    future[:, column] = np.linspace(0, 1, 8, dtype=np.float32)
    target = np.full((8, 256, 256, 3), value + 1, dtype=np.uint8)
    np.savez_compressed(
        path,
        context_frames=context,
        history_actions=history,
        future_actions=future,
        target_frames=target,
    )


def runtime_for(library_dir):
    runtime = Track2V15Runtime.__new__(Track2V15Runtime)
    runtime.library_dir = library_dir
    return runtime


def test_retrieval_source_cache_retains_exact_owned_frames(tmp_path, monkeypatch):
    windows = tmp_path / "windows"
    splits = tmp_path / "splits"
    windows.mkdir()
    splits.mkdir()
    (splits / "split.json").write_text(
        json.dumps({"train_episodes": [1, 2]}), encoding="utf-8"
    )
    make_window(windows / "episode1_00000.npz", 17, 0)
    make_window(windows / "episode2_00000.npz", 29, 1)
    manifest = {
        "retrieval": {
            "split_manifest": "splits/split.json",
            "windows_directory": "windows",
        }
    }

    monkeypatch.setenv("WAM_RETRIEVAL_SOURCE_CACHE", "1")
    cached = runtime_for(tmp_path)
    cached._load_library(manifest)
    assert len(cached.library_sources) == 2
    assert cached.library_sources[0].flags.owndata
    assert np.array_equal(cached._library_source(0), np.full((256, 256, 3), 17, np.uint8))
    assert np.array_equal(cached._library_source(1), np.full((256, 256, 3), 29, np.uint8))

    monkeypatch.setenv("WAM_RETRIEVAL_SOURCE_CACHE", "0")
    uncached = runtime_for(tmp_path)
    uncached._load_library(manifest)
    assert uncached.library_sources is None
    assert np.array_equal(uncached._library_source(0), cached._library_source(0))
    assert np.array_equal(uncached.library_visual, cached.library_visual)
    assert np.array_equal(uncached.library_motion, cached.library_motion)
    assert np.array_equal(uncached.library_arms, cached.library_arms)


def test_retrieval_target_cache_is_lazy_owned_and_exact(tmp_path, monkeypatch):
    windows = tmp_path / "windows"
    splits = tmp_path / "splits"
    windows.mkdir()
    splits.mkdir()
    (splits / "split.json").write_text(
        json.dumps({"train_episodes": [1, 2]}), encoding="utf-8"
    )
    make_window(windows / "episode1_00000.npz", 17, 0)
    make_window(windows / "episode2_00000.npz", 29, 1)
    manifest = {
        "retrieval": {
            "split_manifest": "splits/split.json",
            "windows_directory": "windows",
        }
    }

    monkeypatch.setenv("WAM_RETRIEVAL_TARGET_CACHE", "1")
    cached = runtime_for(tmp_path)
    cached._load_library(manifest)
    assert cached.library_targets == {}
    first = cached._library_target(0)
    assert first.flags.owndata
    assert list(cached.library_targets) == [0]
    assert cached._library_target(0) is first

    monkeypatch.setenv("WAM_RETRIEVAL_TARGET_CACHE", "0")
    uncached = runtime_for(tmp_path)
    uncached._load_library(manifest)
    assert uncached.library_targets is None
    assert np.array_equal(uncached._library_target(0), first)


def test_v141_left_arm_rejects_before_retrieval(tmp_path):
    runtime = runtime_for(tmp_path)
    # No library attributes are installed deliberately: a left-arm request
    # must return before descriptor search, NPZ access, or ECC alignment.
    parent = np.arange(8 * 256 * 256 * 3, dtype=np.uint8).reshape(8, 256, 256, 3)
    context = np.zeros((5, 256, 256, 3), dtype=np.uint8)
    history = np.zeros((4, 14), dtype=np.float32)
    future = np.zeros((8, 14), dtype=np.float32)
    future[:, 0] = np.linspace(0, 1, 8, dtype=np.float32)

    actual = runtime._v141(parent, context, history, future)
    assert np.array_equal(actual, parent)
    assert actual is not parent
