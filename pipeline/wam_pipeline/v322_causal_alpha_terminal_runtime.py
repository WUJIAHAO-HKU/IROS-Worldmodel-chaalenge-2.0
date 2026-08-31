"""Action-causal terminal-strength calibration for strict Track 2.

The v317 failure semantics are preserved.  For valid post-grasp right-arm
requests, the terminal successor is selected from frozen public demonstrations
by endpoint plus visual distance, and the already-frozen public action gate can
raise (never lower) the v271 blend strength.  No reward, outcome, policy,
simulator state, seed identity, or evaluation metadata is read at runtime.
"""

from __future__ import annotations

import numpy as np

from .v216_public_knn_blend_runtime import _visual_descriptor
from .v245_clean_progressive_successor_runtime import PREGRASP_ALPHA
from .v317_batched_sparse_failure_terminal_runtime import (
    Track2V317BatchedSparseFailureTerminal,
)


EPISODE_VISUAL_WEIGHT = 0.25
CAUSAL_ALPHA_PROBABILITY_FLOOR = 0.90
CAUSAL_ALPHA_PROBABILITY_CEIL = 0.99


class Track2V322CausalAlphaTerminal(Track2V317BatchedSparseFailureTerminal):
    """Strengthen only high-confidence valid right terminal transitions."""

    def __init__(self, checkpoint_dir, library_index, device="cuda", action_gate=None):
        super().__init__(checkpoint_dir, library_index, device, action_gate)
        self.terminal_episodes = np.asarray(sorted(self.terminal_row), dtype=np.int64)
        self.terminal_rows = np.asarray(
            [self.terminal_row[int(episode)] for episode in self.terminal_episodes],
            dtype=np.int64,
        )
        raw = (
            self.action[self.terminal_rows].reshape(-1, 12, 14)
            * self.action_std
            + self.action_mean
        )
        self.terminal_endpoints = raw[:, -1, 7:13].astype(np.float32)
        self.episode_clean_rows = tuple(
            self.clean_rows[self.row_episode[self.clean_rows] == episode]
            for episode in self.terminal_episodes
        )
        if any(rows.size == 0 for rows in self.episode_clean_rows):
            raise RuntimeError("terminal episode has no clean public rows")
        self.last_terminal_episode = None
        self.last_base_alpha = None
        self.last_causal_alpha = None
        self.last_calibrated_alpha = None

    def _terminal_for_action_context(
        self, context: np.ndarray, future: np.ndarray
    ) -> tuple[int, int]:
        scale = np.maximum(self.action_std[7:13], 1e-6)
        endpoint_distance = (
            ((self.terminal_endpoints - future[-1, 7:13]) / scale) ** 2
        ).mean(axis=1)
        query_visual = _visual_descriptor(context[-1])
        visual_distance = np.asarray(
            [
                ((self.visual[rows] - query_visual) ** 2).mean(axis=1).min()
                for rows in self.episode_clean_rows
            ],
            dtype=np.float64,
        )
        endpoint_normalized = endpoint_distance / max(
            float(np.median(endpoint_distance)), 1e-9
        )
        visual_normalized = visual_distance / max(
            float(np.median(visual_distance)), 1e-9
        )
        score = endpoint_normalized + EPISODE_VISUAL_WEIGHT * visual_normalized
        index = int(np.argmin(score))
        return int(self.terminal_rows[index]), int(self.terminal_episodes[index])

    @staticmethod
    def _causal_alpha(probability: float) -> float:
        return float(
            np.clip(
                (probability - CAUSAL_ALPHA_PROBABILITY_FLOOR)
                / (CAUSAL_ALPHA_PROBABILITY_CEIL - CAUSAL_ALPHA_PROBABILITY_FLOOR),
                0.0,
                1.0,
            )
        )

    def _right_prediction(self, parent, context, history, future):
        base, _ = self._nearest_clean(context, history, future)
        self.last_base_index = base
        if not self._post_grasp(history, future):
            self.last_delta_ratio = None
            self.last_alignment = None
            self.last_progressive_index = None
            self.last_progressive_alpha = 0.0
            self.last_retrieval_index = base
            self.last_terminal_episode = None
            self.last_base_alpha = None
            self.last_causal_alpha = None
            self.last_calibrated_alpha = None
            return self._blend(parent, self._target(base), future, PREGRASP_ALPHA)

        target, episode = self._terminal_for_action_context(context, future)
        base_alpha = self._alpha_with_context(context, history, future, base)
        probability = self._probability(history, future)
        causal_alpha = self._causal_alpha(probability)
        alpha = max(base_alpha, causal_alpha)
        self.last_progressive_index = target
        self.last_progressive_alpha = alpha
        self.last_retrieval_index = target
        self.last_terminal_episode = episode
        self.last_base_alpha = base_alpha
        self.last_causal_alpha = causal_alpha
        self.last_calibrated_alpha = alpha
        return self._blend(parent, self._target(target), future, alpha)
