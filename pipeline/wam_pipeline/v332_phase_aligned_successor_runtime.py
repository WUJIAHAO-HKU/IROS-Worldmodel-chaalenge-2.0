"""Phase-aligned public successor for the v326 right-arm blind spot.

When the frozen v326 repair gate fires and the visually/action matched public
episode has an eligible sustained-success phase, use the already selected base
row instead of pinning every request to a fixed onset or blank terminal.  The
retrieved sequence therefore advances with the query phase.  The two terminal-
ineligible public episodes retain v326's high-reward terminal fallback.
"""

from __future__ import annotations

import numpy as np

from .v245_clean_progressive_successor_runtime import PREGRASP_ALPHA
from .v324_phase_guarded_terminal_runtime import ACTION_PROBABILITY_MIN, BASE_ALPHA_ZERO_MAX
from .v326_blended_phase_terminal_runtime import REPAIR_ALPHA, Track2V326BlendedPhaseTerminal


class Track2V332PhaseAlignedSuccessor(Track2V326BlendedPhaseTerminal):
    """Use the current successful phase row when it is quality-qualified."""

    def _coherent_phase_override(
        self,
        context: np.ndarray,
        history: np.ndarray,
        future: np.ndarray,
        probability: float | None = None,
        failure_signature: str | None = None,
    ) -> bool:
        if not self._post_grasp(history, future):
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

    def _right_prediction(self, parent, context, history, future):
        base, _ = self._nearest_clean(context, history, future)
        self.last_base_index = base
        self.last_phase_aligned = False
        self.last_phase_target_episode = None
        if not self._post_grasp(history, future):
            self.last_delta_ratio = None
            self.last_alignment = None
            self.last_progressive_index = None
            self.last_progressive_alpha = 0.0
            self.last_retrieval_index = base
            self.last_phase_ready = False
            self.last_phase_override = False
            return self._blend(parent, self._target(base), future, PREGRASP_ALPHA)

        baseline_target = self._terminal_for_context(context)
        baseline_alpha = self._alpha_with_context(context, history, future, base)
        probability = self._probability(history, future)
        phase_ready, phase_episode, phase_start, phase_onset = self._phase(base)
        override = (
            baseline_alpha <= BASE_ALPHA_ZERO_MAX
            and probability >= ACTION_PROBABILITY_MIN
            and phase_ready
        )
        target, alpha = baseline_target, baseline_alpha
        if override:
            if self.eligible_episode.get(phase_episode, False):
                target = base
                self.last_phase_aligned = True
                self.last_phase_target_episode = phase_episode
            else:
                target, fallback_episode = self._success_terminal_for_action_context(
                    context, future
                )
                self.last_phase_target_episode = fallback_episode
            alpha = REPAIR_ALPHA
        self.last_progressive_index = target
        self.last_progressive_alpha = alpha
        self.last_retrieval_index = target
        self.last_phase_ready = phase_ready
        self.last_phase_episode = phase_episode
        self.last_phase_start = phase_start
        self.last_phase_onset = phase_onset
        self.last_phase_override = override
        return self._blend(parent, self._target(target), future, alpha)
