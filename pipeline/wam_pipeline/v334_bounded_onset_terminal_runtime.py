"""Bound v333 same-episode onset repair to the immediate onset neighborhood."""

from __future__ import annotations

from .v245_clean_progressive_successor_runtime import PREGRASP_ALPHA
from .v324_phase_guarded_terminal_runtime import ACTION_PROBABILITY_MIN, BASE_ALPHA_ZERO_MAX
from .v326_blended_phase_terminal_runtime import REPAIR_ALPHA
from .v333_hybrid_onset_terminal_runtime import Track2V333HybridOnsetTerminal


ONSET_REPAIR_MAX_OFFSET = 2


class Track2V334BoundedOnsetTerminal(Track2V333HybridOnsetTerminal):
    """Use same-episode onset only within two public rows of phase onset."""

    def _right_prediction(self, parent, context, history, future):
        base, _ = self._nearest_clean(context, history, future)
        self.last_base_index = base
        self.last_same_episode_onset_repair = False
        self.last_target_episode = None
        self.last_phase_offset = None
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
        phase_offset = phase_start - phase_onset
        override = (
            baseline_alpha <= BASE_ALPHA_ZERO_MAX
            and probability >= ACTION_PROBABILITY_MIN
            and phase_ready
        )
        target, alpha = baseline_target, baseline_alpha
        if override:
            bounded_onset = (
                not self.eligible_episode.get(phase_episode, False)
                and 0 <= phase_offset <= ONSET_REPAIR_MAX_OFFSET
            )
            if bounded_onset:
                target = self.onset_row[phase_episode]
                target_episode = phase_episode
                self.last_same_episode_onset_repair = True
            else:
                target, target_episode = self._success_terminal_for_action_context(
                    context, future
                )
            self.last_target_episode = target_episode
            self.last_phase_offset = phase_offset
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
