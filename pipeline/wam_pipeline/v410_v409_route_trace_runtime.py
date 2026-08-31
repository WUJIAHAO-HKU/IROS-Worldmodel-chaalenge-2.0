"""Output-equivalent v409 runtime with outcome-free per-request telemetry."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .v324_phase_guarded_terminal_runtime import ACTION_PROBABILITY_MIN
from .v409_half_contracted_progressive_runtime import (
    Track2V409HalfContractedProgressive,
)


class Track2V410V409RouteTrace(Track2V409HalfContractedProgressive):
    """Trace actual terminal/progressive routing without changing RGB output."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        value = os.environ.get("WAM_V410_TRACE_PATH", "")
        if not value:
            raise RuntimeError("WAM_V410_TRACE_PATH required")
        self.trace_path = Path(value)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._trace_records: list[dict] = []

    def _route(self, context, history, future, arm):
        source_probability = float(self.source_gate.probability(context))
        right = arm == "right"
        source_ready = source_probability >= self.source_threshold
        post_grasp = bool(self._post_grasp(history, future))
        action_probability = float(self._probability(history, future))
        probability_ready = action_probability >= ACTION_PROBABILITY_MIN
        signature = self._signature(history, future, action_probability)
        broad_eligible = bool(right and source_ready and post_grasp and signature is None)
        strict_eligible = bool(broad_eligible and probability_ready)
        phase_row = self._action_phase_base(history, future) if strict_eligible else None
        phase_probability = (
            float(self.continuous_phase_gate.probability(context, history, future))
            if strict_eligible
            else None
        )
        hard_phase_ready = False
        if strict_eligible:
            hard_phase_ready, _, _, _ = self._phase(phase_row)
        self.last_continuous_phase_probability = phase_probability
        self.last_hard_phase_ready = bool(hard_phase_ready and strict_eligible)
        supported = bool(
            strict_eligible
            and phase_probability is not None
            and phase_probability >= self.support_probability_min
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
                "broad_eligible": broad_eligible,
                "strict_eligible": strict_eligible,
                "phase_probability": phase_probability,
                "hard_phase_ready": bool(hard_phase_ready),
                "supported_v400_route": supported,
            }
        )
        continuous = bool(
            strict_eligible
            and phase_probability is not None
            and phase_probability >= self.continuous_phase_gate.threshold
        )
        return continuous, source_probability, phase_row

    def _apply(self, baseline, context, history, future, arm):
        result = super()._apply(baseline, context, history, future, arm)
        record = self._trace_records[-1]
        progressive = bool(self.last_v406_progressive_route)
        terminal = bool(self.last_v406_terminal_precedence)
        record.update(
            {
                "terminal_precedence": terminal,
                "progressive_route": progressive,
                "contracted_alpha": float(self.last_v406_progressive_alpha),
                "progressive_base_row": int(self.last_base_index) if progressive else None,
                "progressive_target_row": int(self.last_progressive_index) if progressive else None,
                "progressive_target_advance": (
                    int(
                        self.row_start[self.last_progressive_index]
                        - self.row_start[self.last_base_index]
                    )
                    if progressive
                    else None
                ),
            }
        )
        return result

    def _flush(self):
        if self._trace_records:
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
