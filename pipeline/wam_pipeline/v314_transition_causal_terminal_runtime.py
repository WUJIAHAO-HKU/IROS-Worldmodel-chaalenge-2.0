"""Transition-causal terminal dynamics for strict Track 2.

Expert-like right requests use mirrored dynamics for frames 1--7 and the
direct v271 terminal frame.  Rejected right requests use mirrored parametric
dynamics but keep the final scene at the observed context state, preventing a
successful retrieval endpoint from being assigned to an invalid action.
"""

from __future__ import annotations

import numpy as np

from .v271_endpoint_calibrated_terminal_runtime import (
    Track2V271EndpointCalibratedTerminal,
)
from .v312_causal_terminal_mirror_runtime import Track2V312CausalTerminalMirror
from .v290_right_closed_mirror_runtime import route_right


class Track2V314TransitionCausalTerminal(Track2V312CausalTerminalMirror):
    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        self.last_right_route = route_right(history_actions, future_actions)
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
        self.last_gate_accepted = probability >= self.gate_threshold
        if self.last_gate_accepted:
            direct = Track2V271EndpointCalibratedTerminal.predict(
                self,
                context_frames,
                history_actions,
                future_actions,
                seed,
                instruction,
            )
            mirrored = self._mirrored_successor(
                context_frames,
                history_actions,
                future_actions,
                seed,
                instruction,
            )
            output = mirrored.copy()
            output[-1] = direct[-1]
            return output
        output = self._mirrored_parametric(
            context_frames,
            history_actions,
            future_actions,
            seed,
            instruction,
        )
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
