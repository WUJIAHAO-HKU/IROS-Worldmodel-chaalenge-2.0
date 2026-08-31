"""Action-conditioned object geometry primitives for the v17 renderer.

The Track-2 action is the pair of absolute 7-DoF joint commands.  RoboTwin's
recorded HDF5 files additionally contain the next-frame end-effector pose and
camera calibration.  This module learns the deterministic forward projection
from an action to three image landmarks (origin, local x axis, local y axis)
without consuming future RGB at inference time.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn


def quaternion_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Convert a SAPIEN-style ``[w, x, y, z]`` quaternion to a rotation."""
    value = np.asarray(quaternion, dtype=np.float64)
    value = value / max(float(np.linalg.norm(value)), 1e-12)
    w, x, y, z = value
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def project_endpose(
    endpose: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    input_size: tuple[int, int] = (320, 240),
    output_size: tuple[int, int] = (256, 256),
    axis_length: float = 0.06,
) -> np.ndarray:
    """Project an end pose into origin/x-axis/y-axis pixels and camera depth.

    RoboTwin windows resize the full 320x240 head image directly to 256x256,
    so x and y deliberately have different scale factors.
    """
    pose = np.asarray(endpose, dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError(f"endpose must be [7], got {pose.shape}")
    rotation = quaternion_matrix_wxyz(pose[3:])
    points = np.stack(
        (
            pose[:3],
            pose[:3] + axis_length * rotation[:, 0],
            pose[:3] + axis_length * rotation[:, 1],
        )
    )
    camera = (np.asarray(extrinsic, dtype=np.float64) @ np.concatenate(
        (points, np.ones((3, 1), dtype=np.float64)), axis=1
    ).T).T
    image = (np.asarray(intrinsic, dtype=np.float64) @ camera.T).T
    image = image[:, :2] / np.clip(image[:, 2:3], 1e-8, None)
    image[:, 0] *= output_size[0] / input_size[0]
    image[:, 1] *= output_size[1] / input_size[1]
    return np.concatenate((image.reshape(-1), camera[:1, 2]), axis=0).astype(np.float32)


def affine_from_landmarks(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a source-to-target affine matrix from projected pose landmarks."""
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if source.shape[-1] < 6 or target.shape[-1] < 6:
        raise ValueError("landmarks require at least six coordinates")
    return cv2.getAffineTransform(source[:6].reshape(3, 2), target[:6].reshape(3, 2))


def warp_layer(value: np.ndarray, matrix: np.ndarray, interpolation: int) -> np.ndarray:
    """Warp a 256x256 RGB image or mask with a source-to-target transform."""
    return cv2.warpAffine(
        value,
        np.asarray(matrix, dtype=np.float32),
        (value.shape[1], value.shape[0]),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def oracle_update(parent: np.ndarray, target: np.ndarray, candidate: np.ndarray,
                  valid: np.ndarray, margin: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Select transported pixels only when they beat the current value.

    This target-aware operation is intentionally restricted to ceiling analysis;
    it must never be used by a deployable evaluator.
    """
    current_error = np.abs(parent.astype(np.float32) - target.astype(np.float32)).mean(2)
    candidate_error = np.abs(candidate.astype(np.float32) - target.astype(np.float32)).mean(2)
    selected = np.asarray(valid, dtype=bool) & (candidate_error + margin < current_error)
    output = parent.copy(); output[selected] = candidate[selected]
    return output, selected


class ActionPoseProjector(nn.Module):
    """Two-arm forward-kinematics surrogate in calibrated image coordinates."""

    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        self.arms = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(7, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, hidden // 2),
                    nn.SiLU(),
                    nn.Linear(hidden // 2, 7),
                )
                for _ in range(2)
            ]
        )

    def forward(self, action: torch.Tensor, arm_id: torch.Tensor) -> torch.Tensor:
        if action.ndim != 2 or action.shape[1] != 7:
            raise ValueError(f"action must be [B,7], got {tuple(action.shape)}")
        arm_id = arm_id.long().reshape(-1)
        if len(arm_id) != len(action):
            raise ValueError("arm_id and action batch sizes differ")
        output = action.new_empty((len(action), 7))
        for arm in (0, 1):
            selected = arm_id == arm
            if selected.any():
                output[selected] = self.arms[arm](action[selected])
        return output


def mask_landmarks(mask: np.ndarray, minimum_pixels: int = 20) -> np.ndarray | None:
    """Represent a 2-D instance by centroid and signed PCA axis endpoints."""
    coordinates = np.argwhere(np.asarray(mask, dtype=bool))[:, ::-1].astype(np.float64)
    if len(coordinates) < minimum_pixels:
        return None
    center = coordinates.mean(0)
    covariance = np.cov((coordinates - center).T) + np.eye(2) * 1e-3
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]; values = values[order]; vectors = vectors[:, order]
    if vectors[0, 0] < 0:
        vectors[:, 0] *= -1
    if vectors[1, 1] < 0:
        vectors[:, 1] *= -1
    first = center + vectors[:, 0] * max(4.0, 1.7 * np.sqrt(values[0]))
    second = center + vectors[:, 1] * max(4.0, 1.7 * np.sqrt(values[1]))
    return np.concatenate((center, first, second, [np.log(len(coordinates))])).astype(np.float32)


class ActionLayerProjector(nn.Module):
    """Independent image-geometry heads for beam, gripper and bottle layers."""

    layer_names = ("beam", "gripper", "bottle")

    def __init__(self, hidden: int = 192) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(7, hidden), nn.SiLU(),
                    nn.Linear(hidden, hidden), nn.SiLU(),
                    nn.Linear(hidden, hidden // 2), nn.SiLU(),
                    nn.Linear(hidden // 2, 7),
                )
                for _ in range(6)
            ]
        )

    def forward(self, action: torch.Tensor, arm_id: torch.Tensor,
                layer_id: torch.Tensor) -> torch.Tensor:
        arm_id = arm_id.long().reshape(-1); layer_id = layer_id.long().reshape(-1)
        head_id = arm_id * 3 + layer_id
        output = action.new_empty((len(action), 7))
        for index, head in enumerate(self.heads):
            selected = head_id == index
            if selected.any():
                output[selected] = head(action[selected])
        return output


class RelativeLayerProjector(nn.Module):
    """Predict per-layer landmark motion from observed geometry and actions."""

    layer_names = ActionLayerProjector.layer_names

    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(22, hidden), nn.SiLU(),
                    nn.Linear(hidden, hidden), nn.SiLU(),
                    nn.Linear(hidden, hidden // 2), nn.SiLU(),
                    nn.Linear(hidden // 2, 7),
                )
                for _ in range(6)
            ]
        )

    def forward(self, features: torch.Tensor, arm_id: torch.Tensor,
                layer_id: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != 22:
            raise ValueError(f"features must be [B,22], got {tuple(features.shape)}")
        head_id = arm_id.long().reshape(-1) * 3 + layer_id.long().reshape(-1)
        output = features.new_empty((len(features), 7))
        for index, head in enumerate(self.heads):
            selected = head_id == index
            if selected.any():
                output[selected] = head(features[selected])
        return output


def denormalize_pose(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
                     arm_id: torch.Tensor) -> torch.Tensor:
    arm_id = arm_id.long().reshape(-1)
    return value * std[arm_id] + mean[arm_id]


def normalize_action(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
                     arm_id: torch.Tensor) -> torch.Tensor:
    arm_id = arm_id.long().reshape(-1)
    return (value - mean[arm_id]) / std[arm_id]
