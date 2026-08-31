"""Public-train-only continuous visual/action terminal-phase gate.

The feature contract is deliberately independent of reward, outcome, request
identity, seed, and evaluation metadata.  Runtime classification reads only
the five RGB context frames and the 4+8 request actions.
"""

from __future__ import annotations

import numpy as np

from .v337_public_recursive_ood_gate import context_quality_features


FEATURE_VERSION = "v389-public-recursive-visual-action-phase-v1"
POOL_GRID = 8


def _pooled_rgb(frame: np.ndarray) -> np.ndarray:
    value = np.asarray(frame, dtype=np.float32)
    if value.shape != (256, 256, 3):
        raise ValueError(f"expected [256,256,3] frame, got {value.shape}")
    block = 256 // POOL_GRID
    return (
        value.reshape(POOL_GRID, block, POOL_GRID, block, 3).mean((1, 3)) / 255.0
    ).reshape(-1)


def phase_features(
    context: np.ndarray, history: np.ndarray, future: np.ndarray
) -> np.ndarray:
    """Return a compact continuous state/action phase descriptor."""
    context = np.asarray(context)
    history = np.asarray(history, dtype=np.float32)
    future = np.asarray(future, dtype=np.float32)
    if context.shape != (5, 256, 256, 3):
        raise ValueError(f"unexpected context shape {context.shape}")
    if history.shape != (4, 14) or future.shape != (8, 14):
        raise ValueError(
            f"unexpected action shapes history={history.shape}, future={future.shape}"
        )

    pooled = np.stack([_pooled_rgb(frame) for frame in context], axis=0)
    visual = np.concatenate(
        (
            pooled[-1],
            pooled.mean(axis=0),
            np.abs(pooled[-1] - pooled[0]),
            np.abs(np.diff(pooled, axis=0)).mean(axis=0),
        )
    )

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
    return np.concatenate(
        (
            context_quality_features(context),
            visual.astype(np.float32, copy=False),
            actions.reshape(-1),
            delta.reshape(-1),
            summary,
        )
    ).astype(np.float32, copy=False)


class PublicRecursivePhaseGate:
    """Frozen linear probability model for imminent terminal phase."""

    def __init__(self, path) -> None:
        with np.load(path, allow_pickle=False) as payload:
            version = str(payload["feature_version"].item())
            if version != FEATURE_VERSION:
                raise RuntimeError(f"unsupported phase feature version: {version}")
            self.mean = payload["feature_mean"].astype(np.float32)
            self.scale = payload["feature_scale"].astype(np.float32)
            self.coefficient = payload["coefficient"].astype(np.float32)
            self.intercept = float(payload["intercept"].item())
            self.threshold = float(payload["threshold"].item())
        expected = phase_features(
            np.zeros((5, 256, 256, 3), dtype=np.uint8),
            np.zeros((4, 14), dtype=np.float32),
            np.zeros((8, 14), dtype=np.float32),
        ).shape
        if not (
            self.mean.shape
            == self.scale.shape
            == self.coefficient.shape
            == expected
        ):
            raise RuntimeError("phase gate dimension mismatch")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.coefficient).all():
            raise RuntimeError("phase gate contains non-finite parameters")
        if not np.all(self.scale > 0):
            raise RuntimeError("phase gate contains nonpositive scales")

    def probability(
        self, context: np.ndarray, history: np.ndarray, future: np.ndarray
    ) -> float:
        feature = phase_features(context, history, future)
        logit = float(((feature - self.mean) / self.scale) @ self.coefficient + self.intercept)
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0))))

    def predicts_terminal_phase(
        self, context: np.ndarray, history: np.ndarray, future: np.ndarray
    ) -> bool:
        return self.probability(context, history, future) >= self.threshold
