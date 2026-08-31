"""Learned production mapping from AgileX joint commands to Bridge-style EE deltas."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def joint_action_features(actions: np.ndarray, previous: np.ndarray | None = None) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError("joint actions must be [T,14]")
    if previous is None:
        previous_values = np.concatenate((actions[:1], actions[:-1]), axis=0)
    else:
        previous = np.asarray(previous, dtype=np.float32).reshape(1, 14)
        previous_values = np.concatenate((previous, actions[:-1]), axis=0)
    return np.concatenate((actions, actions - previous_values), axis=1).astype(np.float32)


def active_arm_identity(actions: np.ndarray) -> float:
    actions = np.asarray(actions, dtype=np.float32)
    delta = np.abs(np.diff(actions, axis=0))
    left = float(delta[:, :7].sum())
    right = float(delta[:, 7:].sum())
    return -1.0 if left >= right else 1.0


class JointActionMapper(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(28, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 7),
        )
        self.register_buffer("input_mean", torch.zeros(28))
        self.register_buffer("input_std", torch.ones(28))
        self.register_buffer("output_mean", torch.zeros(7))
        self.register_buffer("output_std", torch.ones(7))

    def set_statistics(self, inputs: np.ndarray, outputs: np.ndarray) -> None:
        self.input_mean.copy_(torch.from_numpy(inputs.mean(0).astype(np.float32)))
        self.input_std.copy_(torch.from_numpy(np.maximum(inputs.std(0), 1e-5).astype(np.float32)))
        self.output_mean.copy_(torch.from_numpy(outputs.mean(0).astype(np.float32)))
        self.output_std.copy_(torch.from_numpy(np.maximum(outputs.std(0), 1e-5).astype(np.float32)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.input_mean) / self.input_std
        return self.network(normalized) * self.output_std + self.output_mean

    def normalized_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return ((prediction - target) / self.output_std).square().mean()

