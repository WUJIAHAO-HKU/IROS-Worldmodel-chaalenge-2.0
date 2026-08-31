"""Hybrid of v325 same-episode onset and v326 eligible terminal repair.

For the frozen v326 blind-spot branch, terminal-ineligible matched episodes
use their own sustained-success onset (the v325 strength); eligible matched
episodes retain v326's high-reward terminal selection.  All gates, blend
strength, v317 splice behavior, and failure suppression remain unchanged.
"""

from __future__ import annotations

import numpy as np

from .v245_clean_progressive_successor_runtime import PREGRASP_ALPHA
from .v324_phase_guarded_terminal_runtime import ACTION_PROBABILITY_MIN, BASE_ALPHA_ZERO_MAX
from .v326_blended_phase_terminal_runtime import REPAIR_ALPHA, Track2V326BlendedPhaseTerminal


class Track2V333HybridOnsetTerminal(Track2V326BlendedPhaseTerminal):
    """Use same-episode onset only when that episode's terminal is ineligible."""

    def __init__(
        self,
        checkpoint_dir,
        library_index,
        device="cuda",
        action_gate=None,
        phase_gate=None,
    ):
        super().__init__(checkpoint_dir, library_index, device, action_gate, phase_gate)
        self.onset_row = {}
        finite_limit = np.iinfo(np.int32).max
        for episode, onset in self.phase_onset.items():
            if onset >= finite_limit:
                continue
            matches = self.clean_rows[
                (self.row_episode[self.clean_rows] == episode)
                & (self.row_start[self.clean_rows] == onset)
            ]
            if len(matches) != 1:
                raise RuntimeError(f"v333 onset row is not unique for episode {episode}")
            self.onset_row[episode] = int(matches[0])
        self.last_same_episode_onset_repair = False
        self.last_target_episode = None

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
        self.last_same_episode_onset_repair = False
        self.last_target_episode = None
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
            if not self.eligible_episode.get(phase_episode, False):
                target = self.onset_row[phase_episode]
                target_episode = phase_episode
                self.last_same_episode_onset_repair = True
            else:
                target, target_episode = self._success_terminal_for_action_context(
                    context, future
                )
            self.last_target_episode = target_episode
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
