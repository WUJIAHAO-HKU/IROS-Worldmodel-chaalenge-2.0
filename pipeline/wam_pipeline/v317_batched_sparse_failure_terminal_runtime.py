"""Native-batched deployment of the v315 sparse failure semantics."""

from __future__ import annotations

import numpy as np

from .v271_endpoint_calibrated_terminal_runtime import (
    Track2V271EndpointCalibratedTerminal,
)
from .v290_right_closed_mirror_runtime import mirror_actions, mirror_prompt, route_right
from .v315_sparse_failure_terminal_runtime import (
    GRIPPER_CLOSED_MAX,
    Track2V315SparseFailureTerminal,
)


class Track2V317BatchedSparseFailureTerminal(Track2V315SparseFailureTerminal):
    """Keep v315 single-request behavior and batch only parent inference."""

    def predict_batch(
        self,
        context_frames,
        history_actions,
        future_actions,
        seeds,
        instructions,
    ):
        context_frames = np.asarray(context_frames)
        history_actions = np.asarray(history_actions)
        future_actions = np.asarray(future_actions)
        seeds = np.asarray(seeds)
        direct = Track2V271EndpointCalibratedTerminal.predict_batch(
            self,
            context_frames,
            history_actions,
            future_actions,
            seeds,
            instructions,
        )
        routes = np.asarray(
            [
                route_right(history, future)
                for history, future in zip(history_actions, future_actions, strict=True)
            ],
            dtype=bool,
        )
        probabilities = np.asarray(
            [
                self._probability(history, future) if right else np.nan
                for history, future, right in zip(
                    history_actions, future_actions, routes, strict=True
                )
            ],
            dtype=np.float64,
        )
        closed = routes & np.asarray(
            [float(future[:, 13].mean()) <= GRIPPER_CLOSED_MAX for future in future_actions],
            dtype=bool,
        )
        output = direct.copy()
        if closed.any():
            selected = np.flatnonzero(closed)
            mirrored = Track2V271EndpointCalibratedTerminal.predict_batch(
                self,
                np.ascontiguousarray(context_frames[closed, :, :, ::-1, :]),
                mirror_actions(history_actions[closed]),
                mirror_actions(future_actions[closed]),
                seeds[closed],
                [mirror_prompt(instructions[index]) for index in selected],
            )
            mirrored = np.ascontiguousarray(mirrored[:, :, :, ::-1, :])
            output[closed, :-1] = mirrored[:, :-1]

        signatures = [
            self._signature(history, future, float(probability)) if right else None
            for history, future, probability, right in zip(
                history_actions, future_actions, probabilities, routes, strict=True
            )
        ]
        suppressed = np.asarray([value is not None for value in signatures], dtype=bool)
        if suppressed.any():
            output[suppressed, -1] = context_frames[suppressed, -1]
        self.last_right_route = bool(routes.any())
        self.last_gate_probability = (
            float(probabilities[routes][0]) if int(routes.sum()) == 1 else None
        )
        self.last_failure_signature = signatures[int(np.flatnonzero(routes)[0])] if int(routes.sum()) == 1 else None
        self.last_failure_suppressed = bool(suppressed.any())
        self.last_gate_accepted = bool((routes & ~suppressed).any())
        return output
