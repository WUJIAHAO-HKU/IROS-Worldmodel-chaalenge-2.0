"""Repair terminal-degraded public episodes without crossing scene identity.

The v323 table can identify an episode that contains a sustained successful
phase even though its final library row has degraded below the success floor.
For such episodes only, v325 replaces a zero-alpha v317 terminal with the
same episode's frozen success-onset row.  Every gate is public-trained and
frozen; runtime reward/outcome access is neither needed nor allowed.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .v245_clean_progressive_successor_runtime import PREGRASP_ALPHA
from .v317_batched_sparse_failure_terminal_runtime import (
    Track2V317BatchedSparseFailureTerminal,
)


ACTION_PROBABILITY_MIN = 0.99
BASE_ALPHA_ZERO_MAX = 1e-8


class Track2V325SameEpisodePhaseRepair(Track2V317BatchedSparseFailureTerminal):
    def __init__(
        self,
        checkpoint_dir,
        library_index,
        device="cuda",
        action_gate=None,
        phase_gate=None,
    ):
        super().__init__(checkpoint_dir, library_index, device, action_gate)
        path = Path(phase_gate or os.environ.get("WAM_V325_PHASE_GATE", ""))
        if not path.is_file():
            raise RuntimeError(f"v325 phase gate is missing: {path}")
        with np.load(path, allow_pickle=False) as values:
            if str(values["format"].item()) != "strict-track2-v323-public-terminal-phase-gate-v1":
                raise RuntimeError("unsupported v325 phase-gate format")
            episodes = values["episode"].astype(np.int64)
            onsets = values["onset_start"].astype(np.int64)
            eligible = values["eligible"].astype(bool)
        if not (len(episodes) == len(onsets) == len(eligible)):
            raise RuntimeError("v325 phase-gate arrays are inconsistent")
        if set(episodes.tolist()) != set(int(value) for value in self.terminal_row):
            raise RuntimeError("v325 phase-gate episodes do not match public library")

        self.phase_gate_path = path
        self.phase_onset = {
            int(episode): int(onset) for episode, onset in zip(episodes, onsets, strict=True)
        }
        self.terminal_eligible = {
            int(episode): bool(value) for episode, value in zip(episodes, eligible, strict=True)
        }
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
                raise RuntimeError(f"v325 onset row is not unique for episode {episode}")
            self.onset_row[episode] = int(matches[0])
        self.repair_episodes = tuple(
            episode
            for episode in sorted(self.onset_row)
            if not self.terminal_eligible[episode]
        )
        if len(self.repair_episodes) < 1:
            raise RuntimeError("v325 has no terminal-degraded repair episode")
        self.last_phase_ready = False
        self.last_phase_episode = None
        self.last_phase_start = None
        self.last_phase_onset = None
        self.last_phase_override = False

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
            and phase_episode in self.repair_episodes
        )
        target, alpha = baseline_target, baseline_alpha
        if override:
            target = self.onset_row[phase_episode]
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
