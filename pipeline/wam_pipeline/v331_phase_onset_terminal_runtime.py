"""Use visible public success-onset frames instead of blank episode tails.

The frozen v323 table defines the first row of a sustained >=0.9 reward phase
using public-training data.  For a v326-authorized repair, v331 prefers that
onset row from the visually matched base episode; only the two terminal-
ineligible training episodes fall back to an action/visual matched eligible
onset.  V317 splice and failure semantics remain unchanged.
"""

from __future__ import annotations

import numpy as np

from .v216_public_knn_blend_runtime import _visual_descriptor
from .v245_clean_progressive_successor_runtime import PREGRASP_ALPHA
from .v324_phase_guarded_terminal_runtime import (
    ACTION_PROBABILITY_MIN,
    BASE_ALPHA_ZERO_MAX,
    EPISODE_VISUAL_WEIGHT,
)
from .v326_blended_phase_terminal_runtime import (
    REPAIR_ALPHA,
    Track2V326BlendedPhaseTerminal,
)


class Track2V331PhaseOnsetTerminal(Track2V326BlendedPhaseTerminal):
    """Replace only a qualified v326 repair target with a visible onset row."""

    def __init__(
        self,
        checkpoint_dir,
        library_index,
        device="cuda",
        action_gate=None,
        phase_gate=None,
    ):
        super().__init__(
            checkpoint_dir, library_index, device, action_gate, phase_gate
        )
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
                raise RuntimeError(f"v331 onset row is not unique for episode {episode}")
            self.onset_row[episode] = int(matches[0])
        self.eligible_onset_episodes = np.asarray(
            [episode for episode in self.success_episodes if int(episode) in self.onset_row],
            dtype=np.int64,
        )
        if len(self.eligible_onset_episodes) != int(len(self.success_episodes)):
            raise RuntimeError("v331 eligible onset table is incomplete")
        self.eligible_onset_rows = np.asarray(
            [self.onset_row[int(episode)] for episode in self.eligible_onset_episodes],
            dtype=np.int64,
        )
        raw = (
            self.action[self.eligible_onset_rows].reshape(-1, 12, 14)
            * self.action_std
            + self.action_mean
        )
        self.eligible_onset_endpoints = raw[:, -1, 7:13].astype(np.float32)
        self.last_onset_same_episode = False
        self.last_onset_episode = None

    def _fallback_onset_for_action_context(self, context, future) -> tuple[int, int]:
        scale = np.maximum(self.action_std[7:13], 1e-6)
        endpoint_distance = (
            ((self.eligible_onset_endpoints - future[-1, 7:13]) / scale) ** 2
        ).mean(axis=1)
        query_visual = _visual_descriptor(context[-1])
        visual_distance = np.asarray(
            [
                ((self.visual[rows] - query_visual) ** 2).mean(axis=1).min()
                for rows in self.success_episode_clean_rows
            ],
            dtype=np.float64,
        )
        score = endpoint_distance / max(float(np.median(endpoint_distance)), 1e-9)
        score += EPISODE_VISUAL_WEIGHT * visual_distance / max(
            float(np.median(visual_distance)), 1e-9
        )
        index = int(np.argmin(score))
        return int(self.eligible_onset_rows[index]), int(self.eligible_onset_episodes[index])

    def _onset_target(self, phase_episode: int, context, future) -> tuple[int, int, bool]:
        if self.eligible_episode.get(phase_episode, False):
            return self.onset_row[phase_episode], phase_episode, True
        row, episode = self._fallback_onset_for_action_context(context, future)
        return row, episode, False

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
        self.last_onset_same_episode = False
        self.last_onset_episode = None
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
            target, onset_episode, same_episode = self._onset_target(
                phase_episode, context, future
            )
            alpha = REPAIR_ALPHA
            self.last_onset_episode = onset_episode
            self.last_onset_same_episode = same_episode
        self.last_progressive_index = target
        self.last_progressive_alpha = alpha
        self.last_retrieval_index = target
        self.last_phase_ready = phase_ready
        self.last_phase_episode = phase_episode
        self.last_phase_start = phase_start
        self.last_phase_onset = phase_onset
        self.last_phase_override = override
        return self._blend(parent, self._target(target), future, alpha)
