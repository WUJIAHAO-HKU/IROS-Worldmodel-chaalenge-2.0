"""Sparse public terminal overlay on the frozen v355 parametric dynamics."""

from __future__ import annotations

import numpy as np

from .v315_sparse_failure_terminal_runtime import GRIPPER_CLOSED_MAX
from .v324_phase_guarded_terminal_runtime import (
    ACTION_PROBABILITY_MIN,
    Track2V324PhaseGuardedTerminal,
)
from .v290_right_closed_mirror_runtime import route_right


class Track2V367SparseTerminalOverlay(Track2V324PhaseGuardedTerminal):
    """Leave v355 unchanged except for phase-qualified success/failure terminals."""

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        parent = self.parent.predict(
            context_frames, history_actions, future_actions, seed, instruction
        )
        arm = "right" if route_right(history_actions, future_actions) else "left"
        self.last_arm_route = arm
        self.last_phase_override = False
        self.last_failure_signature = None
        self.last_failure_suppressed = False
        if arm != "right":
            return parent

        probability = self._probability(history_actions, future_actions)
        self.last_gate_probability = probability
        signature = self._signature(history_actions, future_actions, probability)
        self.last_failure_signature = signature
        if signature is not None:
            output = parent.copy()
            output[-1] = context_frames[-1]
            self.last_failure_suppressed = True
            self.last_gate_accepted = False
            return output

        self.last_gate_accepted = True
        if not self._post_grasp(history_actions, future_actions):
            return parent
        base, _ = self._nearest_clean(
            context_frames, history_actions, future_actions
        )
        phase_ready, episode, start, onset = self._phase(base)
        self.last_base_index = base
        self.last_phase_ready = phase_ready
        self.last_phase_episode = episode
        self.last_phase_start = start
        self.last_phase_onset = onset
        if probability < ACTION_PROBABILITY_MIN or not phase_ready:
            return parent
        target, _ = self._success_terminal_for_action_context(
            context_frames, future_actions
        )
        self.last_phase_override = True
        self.last_retrieval_index = target
        self.last_progressive_index = target
        self.last_progressive_alpha = 1.0
        return self._blend(parent, self._target(target), future_actions, 1.0)

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
