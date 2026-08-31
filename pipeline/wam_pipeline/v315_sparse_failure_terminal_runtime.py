"""Sparse action-causal terminal correction for strict Track 2.

The validated v295 path is preserved for every ordinary request.  Only three
explicit invalid right-arm signatures replace the terminal RGB frame with the
observed context: release after a closed grasp, zero-motion transport, or a
closed moving request whose frozen public-train action-gate probability is
below 0.01.  No reward, outcome, policy, or simulator state is read.
"""

from __future__ import annotations

import numpy as np

from .v271_endpoint_calibrated_terminal_runtime import (
    Track2V271EndpointCalibratedTerminal,
)
from .v312_causal_terminal_mirror_runtime import Track2V312CausalTerminalMirror
from .v290_right_closed_mirror_runtime import route_right


REVERSE_LIKE_PROBABILITY_MAX = 0.01
GRIPPER_CLOSED_MAX = 0.5
STATIC_PATH_MAX = 1e-6


class Track2V315SparseFailureTerminal(Track2V312CausalTerminalMirror):
    def _signature(self, history: np.ndarray, future: np.ndarray, probability: float) -> str | None:
        history_gripper = float(history[-1, 13])
        future_gripper = float(future[:, 13].mean())
        sequence = np.concatenate((history[-1:, 7:13], future[:, 7:13]), axis=0)
        path = float(np.linalg.norm(np.diff(sequence, axis=0), axis=1).sum())
        if history_gripper <= GRIPPER_CLOSED_MAX and future_gripper > GRIPPER_CLOSED_MAX:
            return "release_after_closed"
        if path <= STATIC_PATH_MAX:
            return "static_transport"
        if (
            future_gripper <= GRIPPER_CLOSED_MAX
            and probability < REVERSE_LIKE_PROBABILITY_MAX
        ):
            return "reverse_like"
        return None

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        self.last_right_route = route_right(history_actions, future_actions)
        self.last_failure_signature = None
        self.last_failure_suppressed = False
        if not self.last_right_route:
            self.last_gate_probability = None
            self.last_gate_accepted = False
            return Track2V271EndpointCalibratedTerminal.predict(
                self,
                context_frames,
                history_actions,
                future_actions,
                seed,
                instruction,
            )

        probability = self._probability(history_actions, future_actions)
        self.last_gate_probability = probability
        direct = Track2V271EndpointCalibratedTerminal.predict(
            self,
            context_frames,
            history_actions,
            future_actions,
            seed,
            instruction,
        )
        if float(future_actions[:, 13].mean()) <= GRIPPER_CLOSED_MAX:
            mirrored = self._mirrored_successor(
                context_frames,
                history_actions,
                future_actions,
                seed,
                instruction,
            )
            output = mirrored.copy()
            output[-1] = direct[-1]
        else:
            output = direct

        signature = self._signature(history_actions, future_actions, probability)
        self.last_failure_signature = signature
        self.last_failure_suppressed = signature is not None
        self.last_gate_accepted = not self.last_failure_suppressed
        if self.last_failure_suppressed:
            output = output.copy()
            output[-1] = context_frames[-1]
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
