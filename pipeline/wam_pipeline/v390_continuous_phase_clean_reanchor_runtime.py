"""High-specificity continuous-phase replacement for the v385 hard phase gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .v324_phase_guarded_terminal_runtime import ACTION_PROBABILITY_MIN
from .v385_native_batch_clean_reanchor_runtime import (
    Track2V385NativeBatchCleanReanchor,
)
from .v389_public_recursive_phase_gate import PublicRecursivePhaseGate


FORMAT = "strict-track2-v390-continuous-phase-clean-reanchor-release-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Track2V390ContinuousPhaseCleanReanchor(Track2V385NativeBatchCleanReanchor):
    """Preserve v385 except for its diagnosed nearest-row onset boolean."""

    def __init__(self, checkpoint_dir, library_index, device="cuda"):
        super().__init__(checkpoint_dir, library_index, device)
        manifest_path = self.root / "continuous_phase_clean_reanchor_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format") != FORMAT:
            raise RuntimeError("unsupported v390 manifest")
        gate_path = self.root / manifest["phase_gate"]
        if not gate_path.is_file() or sha256(gate_path) != manifest["phase_gate_sha256"]:
            raise RuntimeError("v390 phase gate hash mismatch")
        if manifest.get("reward_or_outcomes_used") is not False:
            raise RuntimeError("v390 runtime data boundary violation")
        self.continuous_phase_gate = PublicRecursivePhaseGate(gate_path)
        if abs(
            self.continuous_phase_gate.threshold
            - float(manifest["phase_threshold"])
        ) > 1e-7:
            raise RuntimeError("v390 phase threshold drift")
        self.last_continuous_phase_probability = None
        self.last_hard_phase_ready = None

    def _route(self, context, history, future, arm):
        source_probability = float(self.source_gate.probability(context))
        self.last_continuous_phase_probability = None
        self.last_hard_phase_ready = None
        if arm != "right" or source_probability < self.source_threshold:
            return False, source_probability, None
        probability = float(self._probability(history, future))
        if not self._post_grasp(history, future) or probability < ACTION_PROBABILITY_MIN:
            return False, source_probability, None
        if self._signature(history, future, probability) is not None:
            return False, source_probability, None
        base = self._action_phase_base(history, future)
        hard_phase_ready, _, _, _ = self._phase(base)
        phase_probability = float(
            self.continuous_phase_gate.probability(context, history, future)
        )
        self.last_continuous_phase_probability = phase_probability
        self.last_hard_phase_ready = bool(hard_phase_ready)
        return (
            phase_probability >= self.continuous_phase_gate.threshold,
            source_probability,
            base,
        )
