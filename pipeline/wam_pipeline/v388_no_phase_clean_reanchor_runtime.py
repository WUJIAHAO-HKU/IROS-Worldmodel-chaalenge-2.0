"""v385 clean reanchor without the policy-incompatible action-phase gate."""
from __future__ import annotations

import json

from .v324_phase_guarded_terminal_runtime import ACTION_PROBABILITY_MIN
from .v385_native_batch_clean_reanchor_runtime import Track2V385NativeBatchCleanReanchor


FORMAT = "strict-track2-v388-no-phase-clean-reanchor-release-v1"


class Track2V388NoPhaseCleanReanchor(Track2V385NativeBatchCleanReanchor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        manifest = json.loads((self.root / "no_phase_clean_reanchor_manifest.json").read_text())
        if manifest.get("format") != FORMAT:
            raise RuntimeError("unsupported v388 manifest")
        if manifest.get("reward_or_outcomes_used") is not False:
            raise RuntimeError("v388 runtime data boundary violation")

    def _route(self, context, history, future, arm):
        source_probability = float(self.source_gate.probability(context))
        if arm != "right" or source_probability < self.source_threshold:
            return False, source_probability, None
        probability = float(self._probability(history, future))
        if not self._post_grasp(history, future) or probability < ACTION_PROBABILITY_MIN:
            return False, source_probability, None
        if self._signature(history, future, probability) is not None:
            return False, source_probability, None
        return True, source_probability, None
