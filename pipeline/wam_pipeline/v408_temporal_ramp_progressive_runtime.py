"""Linear temporal ramp for v407 progression; terminal frame is unchanged."""

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
from .v407_one_chunk_progressive_runtime import Track2V407OneChunkProgressive


FORMAT = "strict-track2-v408-temporal-ramp-progressive-release-v1"


class Track2V408TemporalRampProgressive(Track2V407OneChunkProgressive):
    """Ramp only the progressive overlay from alpha/8 to alpha."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        manifest = json.loads(
            (self.root / "temporal_ramp_progressive_manifest.json").read_text()
        )
        if manifest.get("format") != FORMAT:
            raise RuntimeError("unsupported v408 manifest")
        parent = self.root / "one_chunk_progressive_manifest.json"
        if hashlib.sha256(parent.read_bytes()).hexdigest() != manifest.get(
            "v407_manifest_sha256"
        ):
            raise RuntimeError("v408 parent manifest hash mismatch")
        if manifest.get("ramp_weights") != [
            0.125,
            0.25,
            0.375,
            0.5,
            0.625,
            0.75,
            0.875,
            1.0,
        ]:
            raise RuntimeError("v408 temporal ramp drift")
        if manifest.get("terminal_frame_alpha_preserved") is not True:
            raise RuntimeError("v408 terminal preservation disabled")
        if manifest.get("reward_or_outcomes_used") is not False:
            raise RuntimeError("v408 data boundary violation")

    @staticmethod
    def _ramp_blend(baseline, target, future, alpha):
        ramp = np.linspace(1.0 / 8.0, 1.0, 8, dtype=np.float32)
        closed = (future[:, 13] < 0.5).astype(np.float32)
        weight = (closed * ramp * float(alpha)).reshape(8, 1, 1, 1)
        return np.clip(
            np.rint(
                (1.0 - weight) * baseline.astype(np.float32)
                + weight * target.astype(np.float32)
            ),
            0,
            255,
        ).astype(np.uint8)

    def _apply(self, baseline, context, history, future, arm):
        terminal, terminal_route, source_probability, phase_row, episode = (
            Track2V400SupportedPosteriorBlend._apply(
                self, baseline, context, history, future, arm
            )
        )
        self.last_v406_terminal_precedence = bool(terminal_route)
        self.last_v406_progressive_route = False
        self.last_v406_progressive_alpha = 0.0
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
        target = self._progressive_row(base)
        alpha = float(
            Track2V245CleanProgressiveSuccessor._alpha(
                self, history, future, base, distance
            )
        )
        self.last_v406_progressive_route = bool(alpha > 0.0)
        self.last_v406_progressive_alpha = alpha
        self.last_base_index = base
        self.last_progressive_index = target
        self.last_progressive_alpha = alpha
        self.last_retrieval_index = target
        output = self._ramp_blend(terminal, self._target(target), future, alpha)
        return (
            output,
            bool(alpha > 0.0),
            source_probability,
            base,
            int(self.row_episode[target]),
        )
