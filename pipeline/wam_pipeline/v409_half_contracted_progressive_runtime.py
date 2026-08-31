"""Half-contracted v407 progression with unchanged protected terminals."""

from __future__ import annotations

import hashlib
import json

from .v245_clean_progressive_successor_runtime import (
    Track2V245CleanProgressiveSuccessor,
)
from .v400_supported_posterior_blend_runtime import (
    Track2V400SupportedPosteriorBlend,
)
from .v407_one_chunk_progressive_runtime import Track2V407OneChunkProgressive


FORMAT = "strict-track2-v409-half-contracted-progressive-release-v1"
CONTRACTION_SCALE = 0.5


class Track2V409HalfContractedProgressive(Track2V407OneChunkProgressive):
    """Bound every progressive overlay to at most one-half."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        manifest = json.loads(
            (self.root / "half_contracted_progressive_manifest.json").read_text()
        )
        if manifest.get("format") != FORMAT:
            raise RuntimeError("unsupported v409 manifest")
        parent = self.root / "one_chunk_progressive_manifest.json"
        if hashlib.sha256(parent.read_bytes()).hexdigest() != manifest.get(
            "v407_manifest_sha256"
        ):
            raise RuntimeError("v409 parent manifest hash mismatch")
        if float(manifest.get("contraction_scale", -1.0)) != CONTRACTION_SCALE:
            raise RuntimeError("v409 contraction drift")
        if manifest.get("reward_or_outcomes_used") is not False:
            raise RuntimeError("v409 data boundary violation")

    @staticmethod
    def _contracted_blend(parent, target, future, alpha):
        return Track2V245CleanProgressiveSuccessor._blend(
            parent, target, future, float(alpha) * CONTRACTION_SCALE
        )

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
        contracted = alpha * CONTRACTION_SCALE
        self.last_v406_progressive_route = bool(contracted > 0.0)
        self.last_v406_progressive_alpha = contracted
        self.last_base_index = base
        self.last_progressive_index = target
        self.last_progressive_alpha = contracted
        self.last_retrieval_index = target
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
