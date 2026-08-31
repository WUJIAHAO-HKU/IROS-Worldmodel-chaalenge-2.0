"""Feature contract for the public-train recursive-context corruption gate."""

from __future__ import annotations

import numpy as np


FEATURE_VERSION = "v337-context-quality-temporal-summary-v1"


def context_quality_features(context: np.ndarray) -> np.ndarray:
    """Return scene-agnostic image-quality and temporal statistics."""
    frames = np.asarray(context, dtype=np.float32) / 255.0
    if frames.shape[0] != 5 or frames.shape[-1] != 3:
        raise ValueError(f"expected [5,H,W,3] context, got {frames.shape}")
    values: list[float] = []
    for frame in frames:
        values.extend(frame.mean(axis=(0, 1)).tolist())
        values.extend(frame.std(axis=(0, 1)).tolist())
        values.extend((frame <= (2.0 / 255.0)).mean(axis=(0, 1)).tolist())
        values.extend((frame >= (253.0 / 255.0)).mean(axis=(0, 1)).tolist())
        dx = np.abs(np.diff(frame, axis=1))
        dy = np.abs(np.diff(frame, axis=0))
        values.extend(dx.mean(axis=(0, 1)).tolist())
        values.extend(dy.mean(axis=(0, 1)).tolist())
        values.extend(dx.std(axis=(0, 1)).tolist())
        values.extend(dy.std(axis=(0, 1)).tolist())
    for delta in np.abs(np.diff(frames, axis=0)):
        values.extend(delta.mean(axis=(0, 1)).tolist())
        values.extend(delta.std(axis=(0, 1)).tolist())
        values.extend(delta.max(axis=(0, 1)).tolist())
    return np.asarray(values, dtype=np.float32)


class PublicRecursiveOODGate:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as payload:
            version = str(payload["feature_version"].item())
            if version != FEATURE_VERSION:
                raise RuntimeError(f"unsupported recursive OOD feature version: {version}")
            self.mean = payload["feature_mean"].astype(np.float32)
            self.scale = payload["feature_scale"].astype(np.float32)
            self.coefficient = payload["coefficient"].astype(np.float32)
            self.intercept = float(payload["intercept"].item())
            self.threshold = float(payload["threshold"].item())
            self.corruption_mae_floor = float(payload["corruption_mae_floor"].item())
        if not (
            self.mean.shape == self.scale.shape == self.coefficient.shape
            and np.all(self.scale > 0)
        ):
            raise RuntimeError("invalid recursive OOD gate arrays")

    def probability(self, context: np.ndarray) -> float:
        feature = context_quality_features(context)
        if feature.shape != self.mean.shape:
            raise RuntimeError("recursive OOD feature dimension drift")
        logit = float(((feature - self.mean) / self.scale) @ self.coefficient + self.intercept)
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0))))

    def predicts_corruption(self, context: np.ndarray) -> bool:
        return self.probability(context) >= self.threshold
