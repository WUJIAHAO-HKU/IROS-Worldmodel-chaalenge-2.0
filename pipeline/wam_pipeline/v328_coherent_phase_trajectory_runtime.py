"""Keep phase-qualified right-arm successor trajectories temporally coherent.

V317 deliberately mirrors the first seven frames of every right/closed chunk
and keeps only the direct right-model terminal.  That is conservative for an
isolated terminal, but a closed-loop caller feeds frames 3..7 into its next
five-frame context.  A v326 success repair therefore produced a mixed context:
four mirrored-parent frames followed by one public-success frame.

V328 changes exactly one branch.  When the frozen v326 phase/action gates
authorize its success repair and the v315 failure gate accepts the request,
the complete eight-frame v326 successor is retained.  Every other request
uses the bit-exact v317 splice and suppression behavior.  Runtime inputs remain
request RGB/actions/instruction plus frozen public-training artifacts only.
"""

from __future__ import annotations

import numpy as np

from .v271_endpoint_calibrated_terminal_runtime import (
    Track2V271EndpointCalibratedTerminal,
)
from .v290_right_closed_mirror_runtime import mirror_actions, mirror_prompt, route_right
from .v315_sparse_failure_terminal_runtime import GRIPPER_CLOSED_MAX
from .v324_phase_guarded_terminal_runtime import (
    ACTION_PROBABILITY_MIN,
    BASE_ALPHA_ZERO_MAX,
)
from .v326_blended_phase_terminal_runtime import Track2V326BlendedPhaseTerminal


class Track2V328CoherentPhaseTrajectory(Track2V326BlendedPhaseTerminal):
    """Retain all eight direct frames only for a valid v326 phase override."""

    def _coherent_phase_override(
        self,
        context: np.ndarray,
        history: np.ndarray,
        future: np.ndarray,
        probability: float | None = None,
        failure_signature: str | None = None,
    ) -> bool:
        if not route_right(history, future) or not self._post_grasp(history, future):
            return False
        if probability is None:
            probability = self._probability(history, future)
        if failure_signature is None:
            failure_signature = self._signature(history, future, probability)
        if failure_signature is not None:
            return False
        base, _ = self._nearest_clean(context, history, future)
        baseline_alpha = self._alpha_with_context(context, history, future, base)
        phase_ready, _, _, _ = self._phase(base)
        return bool(
            baseline_alpha <= BASE_ALPHA_ZERO_MAX
            and probability >= ACTION_PROBABILITY_MIN
            and phase_ready
        )

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        right = route_right(history_actions, future_actions)
        self.last_right_route = right
        self.last_coherent_phase_override = False
        self.last_failure_signature = None
        self.last_failure_suppressed = False
        if not right:
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
        signature = self._signature(history_actions, future_actions, probability)
        self.last_gate_probability = probability
        direct = Track2V271EndpointCalibratedTerminal.predict(
            self,
            context_frames,
            history_actions,
            future_actions,
            seed,
            instruction,
        )
        coherent = self._coherent_phase_override(
            context_frames,
            history_actions,
            future_actions,
            probability,
            signature,
        )
        self.last_coherent_phase_override = coherent
        if (
            float(future_actions[:, 13].mean()) <= GRIPPER_CLOSED_MAX
            and not coherent
        ):
            mirrored = Track2V271EndpointCalibratedTerminal.predict(
                self,
                np.ascontiguousarray(context_frames[:, :, ::-1, :]),
                mirror_actions(history_actions),
                mirror_actions(future_actions),
                seed,
                mirror_prompt(instruction),
            )
            mirrored = np.ascontiguousarray(mirrored[:, :, ::-1, :])
            output = mirrored.copy()
            output[-1] = direct[-1]
        else:
            output = direct

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
        signatures = [
            self._signature(history, future, float(probability)) if right else None
            for history, future, probability, right in zip(
                history_actions, future_actions, probabilities, routes, strict=True
            )
        ]
        closed = routes & np.asarray(
            [float(future[:, 13].mean()) <= GRIPPER_CLOSED_MAX for future in future_actions],
            dtype=bool,
        )
        coherent = np.asarray(
            [
                self._coherent_phase_override(
                    context,
                    history,
                    future,
                    float(probability),
                    signature,
                )
                if is_closed
                else False
                for context, history, future, probability, signature, is_closed in zip(
                    context_frames,
                    history_actions,
                    future_actions,
                    probabilities,
                    signatures,
                    closed,
                    strict=True,
                )
            ],
            dtype=bool,
        )

        output = direct.copy()
        splice = closed & ~coherent
        if splice.any():
            selected = np.flatnonzero(splice)
            mirrored = Track2V271EndpointCalibratedTerminal.predict_batch(
                self,
                np.ascontiguousarray(context_frames[splice, :, :, ::-1, :]),
                mirror_actions(history_actions[splice]),
                mirror_actions(future_actions[splice]),
                seeds[splice],
                [mirror_prompt(instructions[index]) for index in selected],
            )
            mirrored = np.ascontiguousarray(mirrored[:, :, :, ::-1, :])
            output[splice, :-1] = mirrored[:, :-1]

        suppressed = np.asarray([value is not None for value in signatures], dtype=bool)
        if suppressed.any():
            output[suppressed, -1] = context_frames[suppressed, -1]
        self.last_right_route = bool(routes.any())
        self.last_gate_probability = (
            float(probabilities[routes][0]) if int(routes.sum()) == 1 else None
        )
        self.last_failure_signature = (
            signatures[int(np.flatnonzero(routes)[0])] if int(routes.sum()) == 1 else None
        )
        self.last_failure_suppressed = bool(suppressed.any())
        self.last_gate_accepted = bool((routes & ~suppressed).any())
        self.last_coherent_phase_override = bool(coherent.any())
        self.last_coherent_phase_override_count = int(coherent.sum())
        return output
