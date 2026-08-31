"""Native-batched, output-equivalent implementation of v295."""

from __future__ import annotations

import re

import numpy as np

from .v271_endpoint_calibrated_terminal_runtime import (
    Track2V271EndpointCalibratedTerminal,
)


MIRROR_SIGN = np.asarray([-1, 1, 1, 1, -1, -1, 1], dtype=np.float32)
RIGHT_GRIPPER_CLOSED_THRESHOLD = 0.5


def mirror_actions(actions: np.ndarray) -> np.ndarray:
    if actions.shape[-1] != 14:
        raise ValueError("Track2 mirror requires 14D actions")
    result = np.empty_like(actions, dtype=np.float32)
    result[..., :7] = actions[..., 7:14] * MIRROR_SIGN
    result[..., 7:14] = actions[..., :7] * MIRROR_SIGN
    return result


def mirror_prompt(prompt: str | None) -> str | None:
    if prompt is None:
        return None
    value = re.sub(r"\bleft\b", "__track2_right__", prompt, flags=re.IGNORECASE)
    value = re.sub(r"\bright\b", "left", value, flags=re.IGNORECASE)
    return value.replace("__track2_right__", "right")


class Track2V301BatchedTerminalFrameMirror(
    Track2V271EndpointCalibratedTerminal
):
    """Same frame rule as v295, using parent-native request batching."""

    def _use_mirror(self, history, future, instruction) -> bool:
        route = self.parent.active_arm(history, future, instruction or "")
        return route == "right" and float(future[:, 13].mean()) <= RIGHT_GRIPPER_CLOSED_THRESHOLD

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        direct = super().predict(
            context_frames, history_actions, future_actions, seed, instruction
        )
        self.last_mirror_applied = self._use_mirror(
            history_actions, future_actions, instruction
        )
        if not self.last_mirror_applied:
            return direct
        mirrored = super().predict(
            np.ascontiguousarray(context_frames[:, :, ::-1, :]),
            mirror_actions(history_actions),
            mirror_actions(future_actions),
            seed,
            mirror_prompt(instruction),
        )
        output = np.ascontiguousarray(mirrored[:, :, ::-1, :])
        output[-1] = direct[-1]
        return output

    def predict_batch(
        self,
        context_frames,
        history_actions,
        future_actions,
        seeds,
        instructions,
    ):
        direct = super().predict_batch(
            context_frames,
            history_actions,
            future_actions,
            seeds,
            instructions,
        )
        mask = np.asarray(
            [
                self._use_mirror(history, future, instruction)
                for history, future, instruction in zip(
                    history_actions, future_actions, instructions, strict=True
                )
            ],
            dtype=bool,
        )
        self.last_mirror_applied = bool(mask.any())
        if not mask.any():
            return direct
        selected = np.flatnonzero(mask)
        mirrored = super().predict_batch(
            np.ascontiguousarray(context_frames[mask, :, :, ::-1, :]),
            mirror_actions(history_actions[mask]),
            mirror_actions(future_actions[mask]),
            np.asarray(seeds)[mask],
            [mirror_prompt(instructions[index]) for index in selected],
        )
        mirrored = np.ascontiguousarray(mirrored[:, :, :, ::-1, :])
        output = direct.copy()
        output[mask, :-1] = mirrored[:, :-1]
        return output
