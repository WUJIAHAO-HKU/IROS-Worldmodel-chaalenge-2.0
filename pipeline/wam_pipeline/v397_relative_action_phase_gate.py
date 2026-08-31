"""Relative-action terminal-phase gate with no RGB or absolute pose inputs."""

from __future__ import annotations

import numpy as np


FEATURE_VERSION = "v397-relative-action-terminal-phase-v1"


def relative_action_features(history: np.ndarray, future: np.ndarray) -> np.ndarray:
    history = np.asarray(history, dtype=np.float32)
    future = np.asarray(future, dtype=np.float32)
    if history.shape != (4, 14) or future.shape != (8, 14):
        raise ValueError(f"unexpected action shapes history={history.shape}, future={future.shape}")
    anchor = history[-1, 7:13]
    relative = future[:, 7:13] - anchor
    sequence = np.concatenate((anchor[None], future[:, 7:13]), axis=0)
    delta = np.diff(sequence, axis=0)
    path = np.linalg.norm(delta, axis=1)
    net = float(np.linalg.norm(relative[-1]))
    summary = np.asarray(
        [path.sum(), net, net / max(float(path.sum()), 1e-8), path.mean(), path.max(),
         np.linalg.norm(np.diff(delta, axis=0), axis=1).mean()], dtype=np.float32
    )
    gripper = np.concatenate((history[-1:, 13], future[:, 13])).astype(np.float32)
    return np.concatenate((relative.reshape(-1), delta.reshape(-1), gripper, summary)).astype(np.float32, copy=False)


class PublicRelativeActionPhaseGate:
    def __init__(self, path) -> None:
        with np.load(path, allow_pickle=False) as payload:
            if str(payload["feature_version"].item()) != FEATURE_VERSION:
                raise RuntimeError("unsupported relative-action phase feature version")
            self.mean = payload["feature_mean"].astype(np.float32)
            self.scale = payload["feature_scale"].astype(np.float32)
            self.coefficient = payload["coefficient"].astype(np.float32)
            self.intercept = float(payload["intercept"].item())
            self.threshold = float(payload["threshold"].item())
        expected = relative_action_features(np.zeros((4, 14), np.float32), np.zeros((8, 14), np.float32)).shape
        if not self.mean.shape == self.scale.shape == self.coefficient.shape == expected:
            raise RuntimeError("relative-action phase gate dimension mismatch")
        if not np.all(self.scale > 0) or not np.isfinite(self.coefficient).all():
            raise RuntimeError("invalid relative-action phase gate parameters")

    def probability(self, history: np.ndarray, future: np.ndarray) -> float:
        feature = relative_action_features(history, future)
        logit = float(((feature - self.mean) / self.scale) @ self.coefficient + self.intercept)
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0))))

    def predicts_terminal_phase(self, history: np.ndarray, future: np.ndarray) -> bool:
        return self.probability(history, future) >= self.threshold
