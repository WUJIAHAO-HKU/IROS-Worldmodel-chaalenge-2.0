"""Reward-monotone real-public-demo successor with v409 blend strength."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .v245_clean_progressive_successor_runtime import (
    Track2V245CleanProgressiveSuccessor,
)
from .v400_supported_posterior_blend_runtime import (
    Track2V400SupportedPosteriorBlend,
)
from .v409_half_contracted_progressive_runtime import (
    CONTRACTION_SCALE,
    Track2V409HalfContractedProgressive,
)


FORMAT = "strict-track2-v417-reward-monotone-successor-release-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Track2V417RewardMonotoneSuccessor(Track2V409HalfContractedProgressive):
    """Select a real future public-demo row with calibrated high reward."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        manifest = json.loads(
            (self.root / "reward_monotone_successor_manifest.json").read_text()
        )
        if manifest.get("format") != FORMAT:
            raise RuntimeError("unsupported v417 manifest")
        parent = self.root / "half_contracted_progressive_manifest.json"
        if sha256(parent) != manifest.get("v409_manifest_sha256"):
            raise RuntimeError("v417 parent manifest hash mismatch")
        value = os.environ.get("WAM_V417_REWARD_MAP", "")
        if not value:
            raise RuntimeError("WAM_V417_REWARD_MAP required")
        reward_map = Path(value)
        if sha256(reward_map) != manifest.get("public_reward_map_sha256"):
            raise RuntimeError("v417 public reward map hash mismatch")
        if manifest.get("max_advance_frames") != 32:
            raise RuntimeError("v417 advance drift")
        if float(manifest.get("minimum_target_reward", -1.0)) != 0.9:
            raise RuntimeError("v417 target threshold drift")
        if manifest.get("runtime_reads_reward_or_outcomes") is not False:
            raise RuntimeError("v417 runtime data boundary violation")
        with np.load(reward_map, allow_pickle=False) as values:
            paths = values["path"].astype(str)
            clean = values["is_clean"].astype(bool)
            reward = values["reward"].astype(np.float32)
        if not np.array_equal(paths, self.paths):
            raise RuntimeError("v417 reward map/library alignment failure")
        if reward.ndim != 3 or reward.shape[0] != len(self.paths) or reward.shape[2] != 8:
            raise RuntimeError("v417 reward map shape failure")
        self.v417_clean = clean
        self.v417_terminal_reward = np.full(len(self.paths), np.nan, dtype=np.float32)
        self.v417_terminal_reward[clean] = np.nanmean(reward[clean, :, -1], axis=1)
        self.v417_max_advance = 32
        self.v417_minimum_reward = 0.9
        self.last_v417_target_reward = None
        self.last_v417_target_advance = None

    def _reward_monotone_row(self, base: int) -> int | None:
        lookup = self.episode_starts[int(self.row_episode[base])]
        base_start = int(self.row_start[base])
        rows = [
            int(lookup[start])
            for start in sorted(lookup)
            if base_start + 8 <= int(start) <= base_start + self.v417_max_advance
        ]
        rows = [
            row
            for row in rows
            if self.v417_clean[row]
            and np.isfinite(self.v417_terminal_reward[row])
            and self.v417_terminal_reward[row] >= self.v417_minimum_reward
        ]
        if not rows:
            return None
        values = np.asarray([self.v417_terminal_reward[row] for row in rows])
        return rows[int(np.argmax(values))]

    def _apply(self, baseline, context, history, future, arm):
        terminal, terminal_route, source_probability, phase_row, episode = (
            Track2V400SupportedPosteriorBlend._apply(
                self, baseline, context, history, future, arm
            )
        )
        self.last_v406_terminal_precedence = bool(terminal_route)
        self.last_v406_progressive_route = False
        self.last_v406_progressive_alpha = 0.0
        self.last_v417_target_reward = None
        self.last_v417_target_advance = None
        if terminal_route:
            return terminal, terminal_route, source_probability, phase_row, episode
        if (
            arm != "right"
            or source_probability < self.source_threshold
            or not self._post_grasp(history, future)
        ):
            return terminal, False, source_probability, phase_row, episode
        action_probability = float(self._probability(history, future))
        if self._signature(history, future, action_probability) is not None:
            return terminal, False, source_probability, phase_row, episode
        base, distance = self._nearest_clean(context, history, future)
        target = self._reward_monotone_row(base)
        self.last_base_index = base
        if target is None:
            self.last_progressive_index = None
            self.last_progressive_alpha = 0.0
            self.last_retrieval_index = base
            return terminal, False, source_probability, base, episode
        alpha = float(
            Track2V245CleanProgressiveSuccessor._alpha(
                self, history, future, base, distance
            )
        )
        contracted = alpha * CONTRACTION_SCALE
        self.last_v406_progressive_route = bool(contracted > 0.0)
        self.last_v406_progressive_alpha = contracted
        self.last_progressive_index = target
        self.last_progressive_alpha = contracted
        self.last_retrieval_index = target
        self.last_v417_target_reward = float(self.v417_terminal_reward[target])
        self.last_v417_target_advance = int(self.row_start[target] - self.row_start[base])
        output = self._contracted_blend(
            terminal, self._target(target), future, alpha
        )
        return (
            output,
            bool(contracted > 0.0),
            source_probability,
            base,
            int(self.row_episode[target]),
        )
