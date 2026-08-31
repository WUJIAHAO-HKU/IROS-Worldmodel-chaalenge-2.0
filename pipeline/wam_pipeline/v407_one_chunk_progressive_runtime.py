"""One official 8-frame action-chunk progression before protected terminals."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .v406_progressive_then_terminal_runtime import (
    Track2V406ProgressiveThenTerminal,
)


FORMAT = "strict-track2-v407-one-chunk-progressive-release-v1"


class Track2V407OneChunkProgressive(Track2V406ProgressiveThenTerminal):
    """Replace v245's 24-frame jump with exactly one 8-frame request horizon."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        manifest = json.loads(
            (self.root / "one_chunk_progressive_manifest.json").read_text()
        )
        if manifest.get("format") != FORMAT:
            raise RuntimeError("unsupported v407 manifest")
        parent = self.root / "progressive_then_terminal_manifest.json"
        if hashlib.sha256(parent.read_bytes()).hexdigest() != manifest.get(
            "v406_manifest_sha256"
        ):
            raise RuntimeError("v407 parent manifest hash mismatch")
        if int(manifest.get("progress_offset_frames", -1)) != 8:
            raise RuntimeError("v407 progress horizon drift")
        if manifest.get("reward_or_outcomes_used") is not False:
            raise RuntimeError("v407 data boundary violation")

    def _progressive_row(self, base: int) -> int:
        lookup = self.episode_starts[int(self.row_episode[base])]
        starts = np.asarray(sorted(lookup), dtype=np.int64)
        goal = int(self.row_start[base]) + 8
        later = starts[starts >= goal]
        chosen = int(later[0] if later.size else starts[-1])
        return int(lookup[chosen])
