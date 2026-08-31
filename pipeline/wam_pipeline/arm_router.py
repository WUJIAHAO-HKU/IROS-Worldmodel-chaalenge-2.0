"""Small deployable arm router for dual-arm Track-2 world-model experts."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


FORMAT = "track2-linear-visual-action-arm-router-v1"


def arm_router_features(
    context_frames: np.ndarray,
    history_actions: np.ndarray,
    future_actions: np.ndarray,
    image_size: int = 12,
) -> np.ndarray:
    """Return the exact visual/action feature used during fitting."""
    image = cv2.resize(
        context_frames[-1], (image_size, image_size), interpolation=cv2.INTER_AREA
    ).astype(np.float32).reshape(-1) / 255.0
    actions = np.concatenate((history_actions, future_actions), axis=0).astype(np.float32)
    delta = np.diff(actions, axis=0)
    action = np.concatenate(
        (
            actions[0],
            actions[-1],
            actions.mean(0),
            actions.std(0),
            delta.mean(0),
            np.abs(delta).mean(0),
            delta.std(0),
            actions.max(0) - actions.min(0),
        )
    )
    return np.concatenate((image, action)).astype(np.float32)


class LinearVisualActionArmRouter:
    """Pure NumPy inference for a standardized logistic arm classifier."""

    def __init__(self, checkpoint_dir: str | Path) -> None:
        root = Path(checkpoint_dir)
        manifest = json.loads((root / "arm_router_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format") != FORMAT:
            raise RuntimeError("unsupported arm-router checkpoint")
        self.image_size = int(manifest["image_size"])
        with np.load(root / "arm_router.npz", allow_pickle=False) as values:
            self.mean = values["mean"].astype(np.float32)
            self.scale = values["scale"].astype(np.float32)
            self.weight = values["weight"].astype(np.float32)
            self.bias = float(values["bias"])
        if not (
            self.mean.shape == self.scale.shape == self.weight.shape
            and np.all(self.scale > 0)
        ):
            raise RuntimeError("invalid arm-router tensor shapes")

    @staticmethod
    def arm_from_instruction(instruction: str | None) -> int | None:
        if not instruction:
            return None
        lowered = instruction.casefold()
        left = "left arm" in lowered
        right = "right arm" in lowered
        if left == right:
            return None
        return int(right)

    def probability_right(self, context_frames, history_actions, future_actions) -> float:
        feature = arm_router_features(
            context_frames, history_actions, future_actions, self.image_size
        )
        if feature.shape != self.mean.shape:
            raise RuntimeError("arm-router input feature shape mismatch")
        logit = float(np.dot((feature - self.mean) / self.scale, self.weight) + self.bias)
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0))))

    def predict(self, context_frames, history_actions, future_actions, instruction=None) -> int:
        explicit = self.arm_from_instruction(instruction)
        if explicit is not None:
            return explicit
        return int(self.probability_right(context_frames, history_actions, future_actions) >= 0.5)
