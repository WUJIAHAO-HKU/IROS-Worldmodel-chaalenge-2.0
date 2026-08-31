"""Reachable public-expert progression with protected v400 terminal semantics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .v245_clean_progressive_successor_runtime import (
    Track2V245CleanProgressiveSuccessor,
)
from .v400_supported_posterior_blend_runtime import (
    Track2V400SupportedPosteriorBlend,
)


FORMAT = "strict-track2-v406-progressive-then-terminal-release-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Track2V406ProgressiveThenTerminal(Track2V400SupportedPosteriorBlend):
    """Use v245 progression before v400's high-specificity terminal route."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        manifest = json.loads(
            (self.root / "progressive_then_terminal_manifest.json").read_text()
        )
        if manifest.get("format") != FORMAT:
            raise RuntimeError("unsupported v406 manifest")
        parent = self.root / "supported_posterior_blend_manifest.json"
        if sha256(parent) != manifest.get("v400_manifest_sha256"):
            raise RuntimeError("v406 parent manifest hash mismatch")
        if manifest.get("progressive_formula") != "frozen v245 unbound _alpha":
            raise RuntimeError("v406 progressive formula drift")
        if manifest.get("terminal_precedence") is not True:
            raise RuntimeError("v406 terminal precedence disabled")
        if manifest.get("reward_or_outcomes_used") is not False:
            raise RuntimeError("v406 data boundary violation")
        self.last_v406_progressive_route = False
        self.last_v406_terminal_precedence = False
        self.last_v406_progressive_alpha = 0.0

    def _apply(self, baseline, context, history, future, arm):
        terminal, terminal_route, source_probability, phase_row, episode = (
            super()._apply(baseline, context, history, future, arm)
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
        # Explicit unbound dispatch avoids v250's later terminal-only _alpha
        # override. This is the exact public v245 reachable-progress formula.
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
        output = self._blend(terminal, self._target(target), future, alpha)
        return (
            output,
            bool(alpha > 0.0),
            source_probability,
            base,
            int(self.row_episode[target]),
        )
