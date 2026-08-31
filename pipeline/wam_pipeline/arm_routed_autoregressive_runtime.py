"""Direct left/right autoregressive parent selected only from request actions."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .autoregressive_unet_runtime import Track2AutoregressiveUNet


class Track2ArmRoutedAutoregressiveUNet:
    def __init__(
        self,
        left_dir: str | Path,
        right_dir: str | Path,
        device: str = "cuda",
    ) -> None:
        self.left = Track2AutoregressiveUNet(left_dir, device)
        self.right = Track2AutoregressiveUNet(right_dir, device)
        self.last_arm_route: str | None = None
        self.last_route: str | None = None

    @staticmethod
    def active_arm(history_actions: np.ndarray, future_actions: np.ndarray,
                   instruction: str = "") -> str:
        text = str(instruction).lower()
        if "right arm" in text:
            return "right"
        if "left arm" in text:
            return "left"
        actions = np.concatenate((history_actions, future_actions), axis=0)
        delta = np.abs(np.diff(actions, axis=0))
        return "right" if delta[:, 7:].mean() > delta[:, :7].mean() else "left"

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        arm = self.active_arm(history_actions, future_actions, instruction)
        self.last_arm_route = arm
        self.last_route = f"candidate_{arm}"
        expert = self.right if arm == "right" else self.left
        return expert.predict(context_frames, history_actions, future_actions, seed, instruction)

    def predict_batch(
        self,
        context_frames: np.ndarray,
        history_actions: np.ndarray,
        future_actions: np.ndarray,
        seeds: np.ndarray,
        instructions: list[str],
    ) -> np.ndarray:
        routes = np.asarray(
            [
                self.active_arm(history, future, instruction)
                for history, future, instruction in zip(
                    history_actions, future_actions, instructions
                )
            ]
        )
        output = np.empty(
            (len(routes), 8, 256, 256, 3), dtype=np.uint8
        )
        for arm, expert in (("left", self.left), ("right", self.right)):
            mask = routes == arm
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
        self.last_arm_route = "mixed" if len(set(routes.tolist())) > 1 else str(routes[0])
        self.last_route = f"candidate_{self.last_arm_route}"
        return output
