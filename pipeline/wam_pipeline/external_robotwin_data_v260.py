"""Streaming access to the public randomized RoboTwin adjust-bottle episodes.

The external files use the ALOHA demonstration layout instead of the Track-2
HDF5 layout.  This module deliberately converts only the head camera and keeps
the exact action/frame alignment used by :mod:`wam_pipeline.data`:
action ``t`` drives the transition from frame ``t`` to frame ``t + 1``.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


CONTEXT_FRAMES = 5
PREDICTION_FRAMES = 8
TOTAL_FRAMES = CONTEXT_FRAMES + PREDICTION_FRAMES
ACTION_DIM = 14
_EPISODE = re.compile(r"episode_(\d+)\.hdf5$")


def episode_identifier(path: Path) -> int:
    match = _EPISODE.search(path.name)
    if match is None:
        raise ValueError(f"not a randomized RoboTwin episode: {path}")
    return int(match.group(1))


def decode_external_rgb(value: object, size: int = 256) -> np.ndarray:
    """Decode the external JPEG bytes with the official RGB resize semantics."""
    if isinstance(value, np.ndarray) and value.dtype == np.uint8:
        raw = value.tobytes()
    elif isinstance(value, (bytes, bytearray, np.bytes_)):
        raw = bytes(value)
    else:
        raise ValueError("external RGB must contain JPEG/PNG bytes")
    with Image.open(io.BytesIO(raw)) as image:
        rgb = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(rgb, dtype=np.uint8).copy()


class ExternalRandomizedWindowDataset(Dataset):
    """Decode uniformly spaced windows without materializing repeated NPZ files."""

    def __init__(
        self,
        root: str | Path,
        *,
        episodes: set[int] | None = None,
        stride: int = 4,
        image_size: int = 256,
    ) -> None:
        if stride < 1 or image_size < 32:
            raise ValueError("stride must be positive and image_size must be at least 32")
        self.root = Path(root)
        self.image_size = int(image_size)
        self.entries: list[tuple[Path, int]] = []
        self.episode_ids: list[int] = []
        paths = sorted(self.root.glob("episode_*/episode_*.hdf5"), key=episode_identifier)
        for path in paths:
            identifier = episode_identifier(path)
            if episodes is not None and identifier not in episodes:
                continue
            try:
                with h5py.File(path, "r") as handle:
                    frame_count = len(handle["observations/images/cam_high"])
                    actions = handle["action"]
                    valid = actions.ndim == 2 and actions.shape == (frame_count, ACTION_DIM)
            except (OSError, KeyError):
                # Snapshot downloads use temporary/incomplete files.  Ignoring
                # those here makes a probe safe while still rejecting an empty set.
                continue
            if not valid or frame_count < TOTAL_FRAMES:
                continue
            starts = list(range(0, frame_count - TOTAL_FRAMES + 1, stride))
            final_start = frame_count - TOTAL_FRAMES
            if starts[-1] != final_start:
                starts.append(final_start)
            self.entries.extend((path, start) for start in starts)
            self.episode_ids.append(identifier)
        if not self.entries:
            raise ValueError(f"no complete external episodes found under {self.root}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        path, start = self.entries[index]
        with h5py.File(path, "r") as handle:
            rgb = handle["observations/images/cam_high"]
            frames = np.stack(
                [decode_external_rgb(rgb[position], self.image_size) for position in range(start, start + TOTAL_FRAMES)]
            )
            actions = np.asarray(handle["action"][start : start + TOTAL_FRAMES - 1], dtype=np.float32)
        if actions.shape != (TOTAL_FRAMES - 1, ACTION_DIM) or not np.isfinite(actions).all():
            raise ValueError(f"malformed external action slice in {path} at {start}")
        return (
            torch.from_numpy(frames[:CONTEXT_FRAMES]),
            torch.from_numpy(actions[: CONTEXT_FRAMES - 1].copy()),
            torch.from_numpy(actions[CONTEXT_FRAMES - 1 :].copy()),
            torch.from_numpy(frames[CONTEXT_FRAMES:]),
        )


def available_external_episodes(root: str | Path) -> list[int]:
    """Return IDs of complete episodes, excluding interrupted downloads."""
    found = ExternalRandomizedWindowDataset(root, stride=10_000)
    return sorted(set(found.episode_ids))
