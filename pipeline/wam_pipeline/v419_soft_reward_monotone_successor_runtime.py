"""Soft v417 successor: dense progress without premature binary success."""

from __future__ import annotations

import hashlib
import json

from .v245_clean_progressive_successor_runtime import (
    Track2V245CleanProgressiveSuccessor,
)
from .v417_reward_monotone_successor_runtime import (
    Track2V417RewardMonotoneSuccessor,
)


FORMAT = "strict-track2-v419-soft-reward-monotone-successor-release-v1"
EFFECTIVE_SCALE = 0.25


class Track2V419SoftRewardMonotoneSuccessor(Track2V417RewardMonotoneSuccessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        manifest = json.loads(
            (self.root / "soft_reward_monotone_successor_manifest.json").read_text()
        )
        if manifest.get("format") != FORMAT:
            raise RuntimeError("unsupported v419 manifest")
        parent = self.root / "reward_monotone_successor_manifest.json"
        if hashlib.sha256(parent.read_bytes()).hexdigest() != manifest.get(
            "v417_manifest_sha256"
        ):
            raise RuntimeError("v419 parent manifest hash mismatch")
        if float(manifest.get("effective_alpha_scale", -1.0)) != EFFECTIVE_SCALE:
            raise RuntimeError("v419 scale drift")

    @staticmethod
    def _contracted_blend(parent, target, future, alpha):
        return Track2V245CleanProgressiveSuccessor._blend(
            parent, target, future, float(alpha) * EFFECTIVE_SCALE
        )

    def _apply(self, *args, **kwargs):
        result = super()._apply(*args, **kwargs)
        if self.last_v406_progressive_route:
            self.last_v406_progressive_alpha *= 0.5
            self.last_progressive_alpha *= 0.5
        return result
