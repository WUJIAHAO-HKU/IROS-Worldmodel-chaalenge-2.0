"""Phase-guarded repair of the v317 right-terminal blind spot.

Only when the original v271 blend is zero, the frozen public action gate is
high-confidence, and a frozen public-train phase table says the matched expert
episode has reached terminal phase, a success-qualified public terminal is
inserted.  All other requests retain v317 semantics exactly.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .v216_public_knn_blend_runtime import _visual_descriptor
from .v245_clean_progressive_successor_runtime import PREGRASP_ALPHA
from .v317_batched_sparse_failure_terminal_runtime import (
    Track2V317BatchedSparseFailureTerminal,
)


EPISODE_VISUAL_WEIGHT = 0.25
ACTION_PROBABILITY_MIN = 0.99
BASE_ALPHA_ZERO_MAX = 1e-8


class Track2V324PhaseGuardedTerminal(Track2V317BatchedSparseFailureTerminal):
    def __init__(
        self,
        checkpoint_dir,
        library_index,
        device="cuda",
        action_gate=None,
        phase_gate=None,
    ):
        super().__init__(checkpoint_dir, library_index, device, action_gate)
        path = Path(phase_gate or os.environ.get("WAM_V324_PHASE_GATE", ""))
        if not path.is_file():
            raise RuntimeError(f"v324 phase gate is missing: {path}")
        with np.load(path, allow_pickle=False) as values:
            if str(values["format"].item()) != "strict-track2-v323-public-terminal-phase-gate-v1":
                raise RuntimeError("unsupported v324 phase-gate format")
            episodes = values["episode"].astype(np.int64)
            onsets = values["onset_start"].astype(np.int64)
            eligible = values["eligible"].astype(bool)
            terminal_reward = values["terminal_reward"].astype(np.float32)
        if not (len(episodes) == len(onsets) == len(eligible) == len(terminal_reward)):
            raise RuntimeError("v324 phase-gate arrays are inconsistent")
        if set(episodes.tolist()) != set(int(value) for value in self.terminal_row):
            raise RuntimeError("v324 phase-gate episodes do not match public library")
        if not np.all(terminal_reward[eligible] >= 0.90):
            raise RuntimeError("v324 eligible terminal below frozen quality floor")
        self.phase_gate_path = path
        self.phase_onset = {
            int(episode): int(onset) for episode, onset in zip(episodes, onsets, strict=True)
        }
        self.eligible_episode = {
            int(episode): bool(value) for episode, value in zip(episodes, eligible, strict=True)
        }
        self.success_episodes = episodes[eligible]
        self.success_terminal_rows = np.asarray(
            [self.terminal_row[int(episode)] for episode in self.success_episodes],
            dtype=np.int64,
        )
        raw = (
            self.action[self.success_terminal_rows].reshape(-1, 12, 14)
            * self.action_std
            + self.action_mean
        )
        self.success_terminal_endpoints = raw[:, -1, 7:13].astype(np.float32)
        self.success_episode_clean_rows = tuple(
            self.clean_rows[self.row_episode[self.clean_rows] == episode]
            for episode in self.success_episodes
        )
        self.last_phase_ready = False
        self.last_phase_episode = None
        self.last_phase_start = None
        self.last_phase_onset = None
        self.last_phase_override = False

    def _success_terminal_for_action_context(self, context, future) -> tuple[int, int]:
        scale = np.maximum(self.action_std[7:13], 1e-6)
        endpoint_distance = (
            ((self.success_terminal_endpoints - future[-1, 7:13]) / scale) ** 2
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
        return int(self.success_terminal_rows[index]), int(self.success_episodes[index])

    def _phase(self, base: int) -> tuple[bool, int, int, int]:
        episode = int(self.row_episode[base])
        start = int(self.row_start[base])
        onset = int(self.phase_onset[episode])
        return start >= onset, episode, start, onset

    def _right_prediction(self, parent, context, history, future):
        base, _ = self._nearest_clean(context, history, future)
        self.last_base_index = base
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
            target, _ = self._success_terminal_for_action_context(context, future)
            alpha = 1.0
        self.last_progressive_index = target
        self.last_progressive_alpha = alpha
        self.last_retrieval_index = target
        self.last_phase_ready = phase_ready
        self.last_phase_episode = phase_episode
        self.last_phase_start = phase_start
        self.last_phase_onset = phase_onset
        self.last_phase_override = override
        return self._blend(parent, self._target(target), future, alpha)
