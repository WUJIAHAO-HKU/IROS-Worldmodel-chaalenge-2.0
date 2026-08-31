"""Track 2 windows for action-conditioned Wan training and evaluation.

The public API has five RGB context frames but only four history actions.
Wan-style sequence models use one action slot per video position, so slot zero is
an explicit zero anchor and never an action from a different time step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .profile import ACTION_DIM, CONTEXT_ACTIONS, CONTEXT_FRAMES, PREDICTION_FRAMES


ANCHOR_ACTION = np.zeros((ACTION_DIM,), dtype=np.float32)
TOTAL_FRAMES = CONTEXT_FRAMES + PREDICTION_FRAMES


@dataclass(frozen=True)
class WanTrack2Window:
    """One exact Track 2 transition, stored without duplicated source frames."""

    context_frames: np.ndarray  # [5, 256, 256, 3] uint8
    history_actions: np.ndarray  # [4, 14] float32
    future_actions: np.ndarray  # [8, 14] float32
    target_frames: np.ndarray  # [8, 256, 256, 3] uint8
    source: str
    start: int

    @property
    def video_frames(self) -> np.ndarray:
        """Return all 13 RGB positions in chronological order."""
        return np.concatenate((self.context_frames, self.target_frames), axis=0)

    @property
    def action_slots(self) -> np.ndarray:
        """Return 13 aligned action slots: anchor, four history, eight future."""
        return np.concatenate(
            (ANCHOR_ACTION[None], self.history_actions, self.future_actions), axis=0
        )


def _episode_directories(root: Path, logical_split: str) -> list[Path]:
    if logical_split not in {"train", "validation"}:
        raise ValueError("logical_split must be train or validation")
    physical_split = "train_data" if logical_split == "train" else "val_data"
    split_root = root / physical_split
    if not split_root.is_dir():
        raise ValueError(f"missing official Wan split directory: {split_root}")
    episodes = [
        episode
        for source in sorted(split_root.iterdir())
        if source.is_dir()
        for episode in sorted(source.iterdir())
        if episode.is_dir()
    ]
    if not episodes:
        raise ValueError(f"no episode directories found under {split_root}")
    return episodes


def _load_episode(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    rgb_path = directory / "rgb.npy"
    actions_path = directory / "actions.npy"
    if not rgb_path.is_file() or not actions_path.is_file():
        raise ValueError(f"missing rgb.npy or actions.npy in {directory}")
    rgb = np.load(rgb_path, mmap_mode="r", allow_pickle=False)
    actions = np.load(actions_path, mmap_mode="r", allow_pickle=False)
    if rgb.dtype != np.uint8 or rgb.ndim != 5 or rgb.shape[1:] != (1, 256, 256, 3):
        raise ValueError(f"invalid RGB sequence in {directory}: {rgb.shape} {rgb.dtype}")
    if actions.dtype != np.float32 or actions.shape != (len(rgb), 1, ACTION_DIM):
        raise ValueError(f"invalid action sequence in {directory}: {actions.shape} {actions.dtype}")
    if not np.isfinite(actions).all():
        raise ValueError(f"non-finite action in {directory}")
    if len(rgb) < TOTAL_FRAMES:
        raise ValueError(f"episode is shorter than one Track 2 window: {directory}")
    return rgb, actions


class WanTrack2WindowDataset:
    """Index non-overlapping source episodes into all valid 5+8 Track 2 windows."""

    def __init__(self, root: str | Path, logical_split: str) -> None:
        self.root = Path(root)
        self.logical_split = logical_split
        self.episodes = _episode_directories(self.root, logical_split)
        self._index: list[tuple[Path, int]] = []
        for directory in self.episodes:
            rgb, _ = _load_episode(directory)
            self._index.extend((directory, start) for start in range(len(rgb) - TOTAL_FRAMES + 1))
        if not self._index:
            raise ValueError("Track 2 dataset has no valid 13-frame windows")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> WanTrack2Window:
        directory, start = self._index[index]
        rgb, actions = _load_episode(directory)
        context_end = start + CONTEXT_FRAMES
        target_end = context_end + PREDICTION_FRAMES
        context_frames = np.asarray(rgb[start:context_end, 0], dtype=np.uint8)
        target_frames = np.asarray(rgb[context_end:target_end, 0], dtype=np.uint8)
        history_actions = np.asarray(
            actions[start : start + CONTEXT_ACTIONS, 0], dtype=np.float32
        )
        future_actions = np.asarray(
            actions[start + CONTEXT_ACTIONS : start + CONTEXT_ACTIONS + PREDICTION_FRAMES, 0],
            dtype=np.float32,
        )
        return WanTrack2Window(
            context_frames=context_frames,
            history_actions=history_actions,
            future_actions=future_actions,
            target_frames=target_frames,
            source=str(directory),
            start=start,
        )
