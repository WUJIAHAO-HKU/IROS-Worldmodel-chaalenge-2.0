"""Output-equivalent v390 runtime with route-stage telemetry only."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .v324_phase_guarded_terminal_runtime import ACTION_PROBABILITY_MIN
from .v390_continuous_phase_clean_reanchor_runtime import (
    Track2V390ContinuousPhaseCleanReanchor,
)


class Track2V391V390RouteTrace(Track2V390ContinuousPhaseCleanReanchor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        value = os.environ.get("WAM_V391_TRACE_PATH", "")
        if not value:
            raise RuntimeError("WAM_V391_TRACE_PATH is required")
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
        base = None
        hard_phase_ready = False
        hard_phase_start = hard_phase_onset = None
        continuous_probability = None
        continuous_ready = False
        if right and source_ready and post_grasp and probability_ready and signature is None:
            base = self._action_phase_base(history, future)
            hard_phase_ready, _, hard_phase_start, hard_phase_onset = self._phase(base)
            continuous_probability = float(
                self.continuous_phase_gate.probability(context, history, future)
            )
            continuous_ready = (
                continuous_probability >= self.continuous_phase_gate.threshold
            )
        route = bool(
            right
            and source_ready
            and post_grasp
            and probability_ready
            and signature is None
            and continuous_ready
        )
        self._trace_records.append(
            {
                "arm": arm,
                "right": right,
                "source_probability": source_probability,
                "source_ready": source_ready,
                "post_grasp": post_grasp,
                "action_probability": action_probability,
                "probability_ready": probability_ready,
                "failure_signature": signature,
                "phase_base_row": base,
                "hard_phase_ready": bool(hard_phase_ready),
                "hard_phase_start": hard_phase_start,
                "hard_phase_onset": hard_phase_onset,
                "continuous_phase_probability": continuous_probability,
                "continuous_phase_ready": bool(continuous_ready),
                "route": route,
            }
        )
        return route, source_probability, base

    def _flush(self):
        if not self._trace_records:
            return
        with self.trace_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps({"batch": self._trace_records}, separators=(",", ":"))
                + "\n"
            )
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
