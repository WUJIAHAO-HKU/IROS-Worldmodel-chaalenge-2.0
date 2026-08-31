"""Mirror right-arm dynamics while preserving v271's final RGB frame exactly."""

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


class Track2V295TerminalFramePreservingMirror(
    Track2V271EndpointCalibratedTerminal
):
    """Use mirrored frames 1--7 and the bit-exact direct v271 frame 8."""

    def _use_mirror(self, history, future, instruction) -> bool:
        route = self.parent.active_arm(history, future, instruction or "")
        self.last_right_route = route == "right"
        self.last_right_gripper_mean = float(future[:, 13].mean())
        return self.last_right_route and self.last_right_gripper_mean <= RIGHT_GRIPPER_CLOSED_THRESHOLD

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        self.last_mirror_applied = self._use_mirror(
            history_actions, future_actions, instruction
        )
        direct = super().predict(
            context_frames, history_actions, future_actions, seed, instruction
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
        return np.stack(
            [
                self.predict(context, history, future, int(seed), instruction)
                for context, history, future, seed, instruction in zip(
                    context_frames,
                    history_actions,
                    future_actions,
                    seeds,
                    instructions,
                    strict=True,
                )
            ],
            axis=0,
        )
