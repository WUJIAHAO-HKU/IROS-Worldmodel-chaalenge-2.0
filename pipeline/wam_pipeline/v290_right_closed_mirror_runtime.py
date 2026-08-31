"""Mirror-equivariant v271 world model for right-arm closed-gripper phases.

This participant-side component transforms only the official world-model
request and returns only predicted RGB frames. It never changes or returns the
organizer's policy actions, rewards, terminations, or success decisions.
"""

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


def route_right(history_actions: np.ndarray, future_actions: np.ndarray) -> bool:
    sequence = np.concatenate([history_actions[-1:], future_actions], axis=0)
    delta = np.abs(np.diff(sequence, axis=0))
    left_motion = float(delta[:, :6].mean())
    right_motion = float(delta[:, 7:13].mean())
    if abs(right_motion - left_motion) > 1e-8:
        return right_motion > left_motion
    # Terminal windows can have no joint motion. The inactive hand stays open,
    # while the active hand remains closed, which resolves the audited tie.
    left_gripper = float(future_actions[:, 6].mean())
    right_gripper = float(future_actions[:, 13].mean())
    return right_gripper < left_gripper - 0.1


def use_right_closed_mirror(
    history_actions: np.ndarray,
    future_actions: np.ndarray,
) -> bool:
    return route_right(history_actions, future_actions) and float(
        future_actions[:, 13].mean()
    ) <= RIGHT_GRIPPER_CLOSED_THRESHOLD


class Track2V290RightClosedMirror(Track2V271EndpointCalibratedTerminal):
    """Use left-equivalent v271 dynamics only for right post-grasp requests."""

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        self.last_right_route = route_right(history_actions, future_actions)
        self.last_right_gripper_mean = float(future_actions[:, 13].mean())
        self.last_mirror_applied = use_right_closed_mirror(history_actions, future_actions)
        if not self.last_mirror_applied:
            return super().predict(
                context_frames, history_actions, future_actions, seed, instruction
            )

        mirrored = super().predict(
            np.ascontiguousarray(context_frames[:, :, ::-1, :]),
            mirror_actions(history_actions),
            mirror_actions(future_actions),
            seed,
            mirror_prompt(instruction),
        )
        return np.ascontiguousarray(mirrored[:, :, ::-1, :])

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
