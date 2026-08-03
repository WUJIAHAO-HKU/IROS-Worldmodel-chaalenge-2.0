"""RoboTwin trajectory loading and 5-context/8-future window construction."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from .profile import ACTION_DIM, CONTEXT_FRAMES, PREDICTION_FRAMES


class DatasetError(ValueError):
    """The source episode cannot satisfy the public Track 2 data contract."""


@dataclass(frozen=True)
class Trajectory:
    """Frame t is paired with absolute action t from the raw dataset."""

    frames: np.ndarray  # [T, H, W, 3] uint8
    actions: np.ndarray  # [T, 14] float32
    source: str


@dataclass(frozen=True)
class Window:
    """An API-aligned 5 context / 8 prediction training sample."""

    context_frames: np.ndarray  # [5, 256, 256, 3] uint8
    history_actions: np.ndarray  # [4, 14]
    future_actions: np.ndarray  # [8, 14]
    target_frames: np.ndarray  # [8, 256, 256, 3] uint8
    source: str
    start: int


def _decode_rgb(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray) and value.ndim == 3:
        array = value
        if array.shape[-1] == 3 and array.dtype == np.uint8:
            return array
    if isinstance(value, np.ndarray) and value.dtype == np.uint8:
        raw = value.tobytes()
    elif isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
    else:
        raise DatasetError("RGB observation must be JPEG/PNG bytes or an HWC uint8 RGB array")
    try:
        with Image.open(io.BytesIO(raw)) as image:
            return np.asarray(image.convert("RGB")).copy()
    except Exception as exc:
        raise DatasetError("failed to decode observation/head_camera/rgb") from exc


def resize_rgb(image: np.ndarray, size: int = 256) -> np.ndarray:
    """Use a single explicit RGB resize path for train and serving data."""
    return np.asarray(Image.fromarray(image, mode="RGB").resize((size, size), Image.Resampling.BILINEAR)).copy()


def load_robotwin_hdf5(path: str | Path, resize_to: int = 256) -> Trajectory:
    """Load the field names documented by the official RL environment guide."""
    path = Path(path)
    try:
        with h5py.File(path, "r") as handle:
            rgb_dataset = handle["observation/head_camera/rgb"]
            actions = np.asarray(handle["joint_action/vector"], dtype=np.float32)
            frames = np.stack([resize_rgb(_decode_rgb(rgb_dataset[index]), resize_to) for index in range(len(rgb_dataset))])
    except KeyError as exc:
        raise DatasetError("expected observation/head_camera/rgb and joint_action/vector") from exc
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise DatasetError(f"joint_action/vector must have shape [T, {ACTION_DIM}]")
    if len(frames) != len(actions):
        raise DatasetError("RGB and action sequences must have the same length")
    if not np.isfinite(actions).all():
        raise DatasetError("action sequence contains NaN or infinity")
    return Trajectory(frames, actions, str(path))


def make_windows(trajectory: Trajectory) -> list[Window]:
    """Build windows with the API alignment: action t drives frame t -> t+1."""
    minimum_frames = CONTEXT_FRAMES + PREDICTION_FRAMES
    if len(trajectory.frames) < minimum_frames:
        raise DatasetError(f"trajectory needs at least {minimum_frames} frames")
    windows: list[Window] = []
    # A window ending at context frame start+4 predicts frames start+5 through start+12.
    for start in range(0, len(trajectory.frames) - minimum_frames + 1):
        history_actions = trajectory.actions[start : start + CONTEXT_FRAMES - 1]
        future_actions = trajectory.actions[start + CONTEXT_FRAMES - 1 : start + CONTEXT_FRAMES - 1 + PREDICTION_FRAMES]
        windows.append(
            Window(
                context_frames=trajectory.frames[start : start + CONTEXT_FRAMES],
                history_actions=history_actions,
                future_actions=future_actions,
                target_frames=trajectory.frames[start + CONTEXT_FRAMES : start + CONTEXT_FRAMES + PREDICTION_FRAMES],
                source=trajectory.source,
                start=start,
            )
        )
    return windows


def write_window_npz(window: Window, path: str | Path) -> None:
    """Store the adapted sample used by lightweight iVideoGPT training and tests."""
    np.savez_compressed(
        path,
        context_frames=window.context_frames,
        history_actions=window.history_actions,
        future_actions=window.future_actions,
        target_frames=window.target_frames,
        source=np.asarray(window.source),
        start=np.asarray(window.start, dtype=np.int64),
    )


def load_window_npz(path: str | Path) -> Window:
    with np.load(path, allow_pickle=False) as data:
        return Window(
            context_frames=np.asarray(data["context_frames"], dtype=np.uint8),
            history_actions=np.asarray(data["history_actions"], dtype=np.float32),
            future_actions=np.asarray(data["future_actions"], dtype=np.float32),
            target_frames=np.asarray(data["target_frames"], dtype=np.uint8),
            source=str(data["source"].item()),
            start=int(data["start"].item()),
        )
