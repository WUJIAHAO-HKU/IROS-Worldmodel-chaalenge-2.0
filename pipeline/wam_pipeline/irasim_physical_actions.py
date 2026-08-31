"""Bridge-compatible physical action conditioning for bimanual RoboTwin data."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


BRIDGE_ACTION_DIM = 8
BRIDGE_ACTION_SCALE = np.asarray([20.0] * 6 + [1.0, 1.0], dtype=np.float32)


@dataclass(frozen=True)
class PhysicalWindow:
    frames: np.ndarray  # [13, 256, 320, 3] uint8
    actions: np.ndarray  # [12, 8] float32: ee delta 7 + arm identity
    active_arm: str
    source: str
    start: int


def _quat_wxyz_to_matrix(value: np.ndarray) -> np.ndarray:
    q = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("invalid end-effector quaternion")
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_euler_zyx(rotation: np.ndarray) -> np.ndarray:
    """Return x/y/z Euler angles for R = Rz @ Ry @ Rx, matching IRASim."""
    sy = math.sqrt(float(rotation[0, 0] ** 2 + rotation[1, 0] ** 2))
    if sy >= 1e-6:
        x = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        y = math.atan2(float(-rotation[2, 0]), sy)
        z = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        x = math.atan2(float(-rotation[1, 2]), float(rotation[1, 1]))
        y = math.atan2(float(-rotation[2, 0]), sy)
        z = 0.0
    return np.asarray([x, y, z], dtype=np.float32)


def _active_arm(joint_actions: np.ndarray) -> str:
    if joint_actions.ndim != 2 or joint_actions.shape[1] != 14:
        raise ValueError("joint_action/vector must be [T,14]")
    left_motion = float(np.abs(np.diff(joint_actions[:, :7], axis=0)).sum())
    right_motion = float(np.abs(np.diff(joint_actions[:, 7:], axis=0)).sum())
    if max(left_motion, right_motion) < 1e-8:
        raise ValueError("episode has no active arm")
    return "left" if left_motion >= right_motion else "right"


def _relative_ee_actions(endpose: np.ndarray, gripper: np.ndarray, arm_identity: float) -> np.ndarray:
    if endpose.ndim != 2 or endpose.shape[1] != 7 or len(endpose) != len(gripper):
        raise ValueError("endpose/gripper shapes are inconsistent")
    result = np.zeros((len(endpose) - 1, BRIDGE_ACTION_DIM), dtype=np.float32)
    for index in range(len(result)):
        previous_position = endpose[index, :3]
        current_position = endpose[index + 1, :3]
        previous_rotation = _quat_wxyz_to_matrix(endpose[index, 3:])
        current_rotation = _quat_wxyz_to_matrix(endpose[index + 1, 3:])
        result[index, :3] = previous_rotation.T @ (current_position - previous_position)
        result[index, 3:6] = _matrix_to_euler_zyx(previous_rotation.T @ current_rotation)
        result[index, 6] = float(gripper[index + 1])
        result[index, 7] = arm_identity
    result *= BRIDGE_ACTION_SCALE
    if not np.isfinite(result).all():
        raise ValueError("physical action conversion produced non-finite values")
    return result


def _decode_frame(value: object, size: tuple[int, int]) -> np.ndarray:
    raw = value.tobytes() if isinstance(value, np.ndarray) else bytes(value)
    with Image.open(io.BytesIO(raw)) as image:
        return np.asarray(image.convert("RGB").resize(size, Image.Resampling.BILINEAR)).copy()


def load_physical_window(window_path: str | Path, native_size: tuple[int, int] = (320, 256)) -> PhysicalWindow:
    """Load native-aspect frames and oracle EE deltas paired with a Track 2 window."""
    window_path = Path(window_path)
    with np.load(window_path, allow_pickle=False) as item:
        source = str(item["source"].item())
        start = int(item["start"].item())
    source_path = Path(source)
    if not source_path.is_file():
        candidate = window_path.parents[1] / "datasets" / "aloha-agilex_clean_50" / "data" / source_path.name
        if candidate.is_file():
            source_path = candidate
        else:
            raise FileNotFoundError(source)
    indices = np.arange(start, start + 13)
    with h5py.File(source_path, "r") as handle:
        joint_actions = np.asarray(handle["joint_action/vector"], dtype=np.float64)
        arm = _active_arm(joint_actions)
        endpose = np.asarray(handle[f"endpose/{arm}_endpose"][indices], dtype=np.float64)
        gripper = np.asarray(handle[f"joint_action/{arm}_gripper"][indices], dtype=np.float64)
        rgb = handle["observation/head_camera/rgb"]
        frames = np.stack([_decode_frame(rgb[int(index)], native_size) for index in indices])
    actions = _relative_ee_actions(endpose, gripper, -1.0 if arm == "left" else 1.0)
    return PhysicalWindow(frames, actions, arm, str(source_path), start)

