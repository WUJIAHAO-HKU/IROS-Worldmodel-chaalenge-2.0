"""One-chunk successor anchored by action phase instead of visual retrieval phase."""

from __future__ import annotations

import hashlib
import json

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


FORMAT = "strict-track2-v420-action-phase-progressive-release-v1"


class Track2V420ActionPhaseProgressive(Track2V409HalfContractedProgressive):
    """Advance exactly eight real frames from the action-only phase match."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        manifest = json.loads(
            (self.root / "action_phase_progressive_manifest.json").read_text()
        )
        if manifest.get("format") != FORMAT:
            raise RuntimeError("unsupported v420 manifest")
        parent = self.root / "half_contracted_progressive_manifest.json"
        if hashlib.sha256(parent.read_bytes()).hexdigest() != manifest.get(
            "v409_manifest_sha256"
        ):
            raise RuntimeError("v420 parent manifest hash mismatch")
        if manifest.get("progress_offset_frames") != 8:
            raise RuntimeError("v420 progress offset drift")
        if manifest.get("phase_anchor") != "nearest clean normalized action sequence":
            raise RuntimeError("v420 phase anchor drift")
        if manifest.get("reward_or_outcomes_used") is not False:
            raise RuntimeError("v420 data boundary violation")
        self.last_v420_phase_base = None
        self.last_v420_phase_start = None

    def _apply(self, baseline, context, history, future, arm):
        terminal, terminal_route, source_probability, phase_row, episode = (
            Track2V400SupportedPosteriorBlend._apply(
                self, baseline, context, history, future, arm
            )
        )
        self.last_v406_terminal_precedence = bool(terminal_route)
        self.last_v406_progressive_route = False
        self.last_v406_progressive_alpha = 0.0
        self.last_v420_phase_base = None
        self.last_v420_phase_start = None
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
        base = self._action_phase_base(history, future)
        actions = np.concatenate((history, future), axis=0).astype(np.float32)
        query = ((actions - self.action_mean) / self.action_std).reshape(-1)
        distance = float(((self.action[base] - query) ** 2).mean())
        target = self._progressive_row(base)
        alpha = float(
            Track2V245CleanProgressiveSuccessor._alpha(
                self, history, future, base, distance
            )
        )
        contracted = alpha * CONTRACTION_SCALE
        self.last_v406_progressive_route = bool(contracted > 0.0)
        self.last_v406_progressive_alpha = contracted
        self.last_base_index = base
        self.last_progressive_index = target
        self.last_progressive_alpha = contracted
        self.last_retrieval_index = target
        self.last_v420_phase_base = base
        self.last_v420_phase_start = int(self.row_start[base])
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
