"""Action-only motion guard for the v423 learned right-arm dynamics expert."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .arm_routed_autoregressive_runtime import Track2ArmRoutedAutoregressiveUNet
from .autoregressive_unet_runtime import Track2AutoregressiveUNet


RIGHT_NORMALIZED_MOTION_MAX = 0.03


class Track2V426MotionGuardedRight:
    """Use v423 on low/moderate right motion and frozen v208 on high motion.

    Routing consumes only instruction and joint14 request actions.  Context,
    seed, reward, request identity, and evaluation outcomes never affect it.
    """

    def __init__(
        self,
        left_dir: str | Path,
        learned_right_dir: str | Path,
        frozen_right_dir: str | Path,
        device: str = "cuda",
        right_motion_max: float = RIGHT_NORMALIZED_MOTION_MAX,
    ) -> None:
        if right_motion_max <= 0.0:
            raise ValueError("right motion threshold must be positive")
        self.left = Track2AutoregressiveUNet(left_dir, device)
        self.learned_right = Track2AutoregressiveUNet(learned_right_dir, device)
        self.frozen_right = Track2AutoregressiveUNet(frozen_right_dir, device)
        self.right_motion_max = float(right_motion_max)
        self.motion_scale = self.learned_right.std.copy()
        self.last_arm_route: str | None = None
        self.last_route: str | None = None

    @staticmethod
    def active_arm(history_actions: np.ndarray, future_actions: np.ndarray, instruction: str = "") -> str:
        return Track2ArmRoutedAutoregressiveUNet.active_arm(history_actions, future_actions, instruction)

    def right_motion(self, history_actions: np.ndarray, future_actions: np.ndarray) -> float:
        actions = np.concatenate((history_actions, future_actions), axis=0).astype(np.float32)
        normalized_delta = np.abs(np.diff(actions, axis=0)) / self.motion_scale
        return float(normalized_delta[:, 7:13].mean())

    def route(self, history_actions: np.ndarray, future_actions: np.ndarray, instruction: str) -> str:
        arm = self.active_arm(history_actions, future_actions, instruction)
        if arm == "left":
            return "left"
        return "learned_right" if self.right_motion(history_actions, future_actions) <= self.right_motion_max else "frozen_right"

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        route = self.route(history_actions, future_actions, instruction)
        self.last_arm_route = "left" if route == "left" else "right"
        self.last_route = route
        expert = self.left if route == "left" else self.learned_right if route == "learned_right" else self.frozen_right
        return expert.predict(context_frames, history_actions, future_actions, seed, instruction)

    def predict_batch(
        self,
        context_frames: np.ndarray,
        history_actions: np.ndarray,
        future_actions: np.ndarray,
        seeds: np.ndarray,
        instructions: list[str],
    ) -> np.ndarray:
        routes = np.asarray([
            self.route(history, future, instruction)
            for history, future, instruction in zip(history_actions, future_actions, instructions)
        ])
        output = np.empty((len(routes), 8, 256, 256, 3), dtype=np.uint8)
        experts = (
            ("left", self.left),
            ("learned_right", self.learned_right),
            ("frozen_right", self.frozen_right),
        )
        for route, expert in experts:
            mask = routes == route
            if not mask.any():
                continue
            selected = np.flatnonzero(mask)
            output[mask] = expert.predict_batch(
                context_frames[mask],
                history_actions[mask],
                future_actions[mask],
                np.asarray(seeds)[mask],
                [instructions[index] for index in selected],
            )
        self.last_arm_route = "mixed"
        self.last_route = "mixed" if len(set(routes.tolist())) > 1 else str(routes[0])
        return output
