"""Action-only terminal-phase features immune to recursive RGB domain shift."""

from __future__ import annotations

import numpy as np


FEATURE_VERSION = "v396-action-only-terminal-phase-v1"


def action_phase_features(history: np.ndarray, future: np.ndarray) -> np.ndarray:
    history = np.asarray(history, dtype=np.float32)
    future = np.asarray(future, dtype=np.float32)
    if history.shape != (4, 14) or future.shape != (8, 14):
        raise ValueError(f"unexpected action shapes history={history.shape}, future={future.shape}")
    actions = np.concatenate((history, future), axis=0)
    delta = np.diff(actions, axis=0)
    right_sequence = np.concatenate((history[-1:, 7:13], future[:, 7:13]), axis=0)
    right_delta = np.diff(right_sequence, axis=0)
    right_path = np.linalg.norm(right_delta, axis=1)
    right_relative = future[:, 7:13] - history[-1, 7:13]
    summary = np.concatenate(
        (
            actions.mean(axis=0),
            actions.std(axis=0),
            actions.min(axis=0),
            actions.max(axis=0),
            np.abs(delta).mean(axis=0),
            np.abs(delta).max(axis=0),
            right_relative[-1],
            np.asarray(
                [
                    right_path.sum(),
                    right_path.mean(),
                    right_path.max(),
                    np.linalg.norm(right_relative[-1]),
                    float(history[-1, 13]),
                    float(future[:, 13].mean()),
                    float((future[:, 13] < 0.5).mean()),
                ],
                dtype=np.float32,
            ),
        )
    )
    return np.concatenate((actions.reshape(-1), delta.reshape(-1), summary)).astype(
        np.float32, copy=False
    )


class PublicActionPhaseGate:
    def __init__(self, path) -> None:
        with np.load(path, allow_pickle=False) as payload:
            version = str(payload["feature_version"].item())
            if version != FEATURE_VERSION:
                raise RuntimeError(f"unsupported action phase feature version: {version}")
            self.mean = payload["feature_mean"].astype(np.float32)
            self.scale = payload["feature_scale"].astype(np.float32)
            self.coefficient = payload["coefficient"].astype(np.float32)
            self.intercept = float(payload["intercept"].item())
            self.threshold = float(payload["threshold"].item())
        expected = action_phase_features(
            np.zeros((4, 14), dtype=np.float32),
            np.zeros((8, 14), dtype=np.float32),
        ).shape
        if not self.mean.shape == self.scale.shape == self.coefficient.shape == expected:
            raise RuntimeError("action phase gate dimension mismatch")
        if not np.all(self.scale > 0) or not np.isfinite(self.coefficient).all():
            raise RuntimeError("invalid action phase gate parameters")

    def probability(self, history: np.ndarray, future: np.ndarray) -> float:
        feature = action_phase_features(history, future)
        logit = float(((feature - self.mean) / self.scale) @ self.coefficient + self.intercept)
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0))))

    def predicts_terminal_phase(self, history: np.ndarray, future: np.ndarray) -> bool:
        return self.probability(history, future) >= self.threshold
