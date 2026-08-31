"""Output-equivalent v385 runtime with public-rollout route telemetry."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .v324_phase_guarded_terminal_runtime import ACTION_PROBABILITY_MIN
from .v385_native_batch_clean_reanchor_runtime import Track2V385NativeBatchCleanReanchor


class Track2V387V385RouteTrace(Track2V385NativeBatchCleanReanchor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        value = os.environ.get("WAM_V387_TRACE_PATH", "")
        if not value:
            raise RuntimeError("WAM_V387_TRACE_PATH is required")
        self.trace_path = Path(value)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._trace_records = []

    def _route(self, context, history, future, arm):
        source_probability = float(self.source_gate.probability(context))
        right = arm == "right"
        source_ready = source_probability >= self.source_threshold
        post_grasp = bool(self._post_grasp(history, future))
        action_probability = float(self._probability(history, future))
        probability_ready = action_probability >= ACTION_PROBABILITY_MIN
        signature = self._signature(history, future, action_probability)
        phase_ready = False
        phase_episode = phase_start = phase_onset = None
        base = None
        if right and source_ready and post_grasp and probability_ready and signature is None:
            base = self._action_phase_base(history, future)
            phase_ready, phase_episode, phase_start, phase_onset = self._phase(base)
        route = bool(right and source_ready and post_grasp and probability_ready and signature is None and phase_ready)
        self._trace_records.append({
            "arm": arm, "right": right,
            "source_probability": source_probability, "source_ready": source_ready,
            "post_grasp": post_grasp, "action_probability": action_probability,
            "probability_ready": probability_ready, "failure_signature": signature,
            "phase_ready": bool(phase_ready), "phase_base_row": base,
            "phase_episode": phase_episode, "phase_start": phase_start,
            "phase_onset": phase_onset, "route": route,
        })
        return route, source_probability, base

    def _flush(self):
        if not self._trace_records:
            return
        with self.trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"batch": self._trace_records}, separators=(",", ":")) + "\n")
        self._trace_records = []

    def predict(self, *args, **kwargs):
        self._trace_records = []
        output = super().predict(*args, **kwargs)
        self._flush()
        return output

    def predict_batch(self, *args, **kwargs):
        self._trace_records = []
        output = super().predict_batch(*args, **kwargs)
        self._flush()
        return output
