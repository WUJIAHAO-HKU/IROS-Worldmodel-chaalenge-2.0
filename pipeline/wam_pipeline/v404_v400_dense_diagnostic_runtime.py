"""Output-equivalent v400 runtime with dense, outcome-free route telemetry."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .v324_phase_guarded_terminal_runtime import ACTION_PROBABILITY_MIN
from .v400_supported_posterior_blend_runtime import (
    Track2V400SupportedPosteriorBlend,
)


class Track2V404V400DenseDiagnostic(Track2V400SupportedPosteriorBlend):
    """Measure rejected right post-grasp requests without changing v400 RGB."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        value = os.environ.get("WAM_V404_TRACE_PATH", "")
        if not value:
            raise RuntimeError("WAM_V404_TRACE_PATH required")
        self.trace_path = Path(value)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._trace_records = []

    def _endpoint_metrics(self, history, future):
        scale = np.maximum(self.action_std[7:13], 1e-6)
        endpoint_distance = np.mean(
            ((self.success_terminal_endpoints - future[-1, 7:13]) / scale) ** 2,
            axis=1,
        )
        endpoint_index = int(np.argmin(endpoint_distance))
        sequence = np.concatenate((history[-1:, 7:13], future[:, 7:13]), axis=0)
        delta = np.diff(sequence, axis=0)
        return {
            "nearest_endpoint_distance": float(endpoint_distance[endpoint_index]),
            "nearest_endpoint_episode": int(self.success_episodes[endpoint_index]),
            "chunk_path_length": float(np.linalg.norm(delta, axis=1).sum()),
            "chunk_net_motion": float(np.linalg.norm(sequence[-1] - sequence[0])),
        }

    def _route(self, context, history, future, arm):
        source_probability = float(self.source_gate.probability(context))
        right = arm == "right"
        source_ready = source_probability >= self.source_threshold
        post_grasp = bool(self._post_grasp(history, future))
        action_probability = float(self._probability(history, future))
        probability_ready = action_probability >= ACTION_PROBABILITY_MIN
        signature = self._signature(history, future, action_probability)
        broad_eligible = bool(
            right and source_ready and post_grasp and signature is None
        )
        strict_eligible = bool(broad_eligible and probability_ready)

        dense_phase_probability = None
        dense_phase_base_row = None
        hard_phase_ready = False
        endpoint = {
            "nearest_endpoint_distance": None,
            "nearest_endpoint_episode": None,
            "chunk_path_length": None,
            "chunk_net_motion": None,
        }
        if broad_eligible:
            dense_phase_base_row = self._action_phase_base(history, future)
            hard_phase_ready, _, _, _ = self._phase(dense_phase_base_row)
            dense_phase_probability = float(
                self.continuous_phase_gate.probability(context, history, future)
            )
            endpoint = self._endpoint_metrics(history, future)

        # Preserve v400 semantics exactly: the phase row/probability is visible
        # to _apply only after the original action-probability >= 0.99 gate.
        phase_row = dense_phase_base_row if strict_eligible else None
        runtime_phase_probability = (
            dense_phase_probability if strict_eligible else None
        )
        self.last_continuous_phase_probability = runtime_phase_probability
        self.last_hard_phase_ready = bool(hard_phase_ready and strict_eligible)
        supported = bool(
            strict_eligible
            and runtime_phase_probability is not None
            and runtime_phase_probability >= self.support_probability_min
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
                "dense_phase_base_row": dense_phase_base_row,
                "dense_phase_probability": dense_phase_probability,
                "hard_phase_ready": bool(hard_phase_ready),
                "supported_v400_route": supported,
                **endpoint,
            }
        )
        continuous = bool(
            strict_eligible
            and runtime_phase_probability is not None
            and runtime_phase_probability >= self.continuous_phase_gate.threshold
        )
        return continuous, source_probability, phase_row

    def _flush(self):
        if self._trace_records:
            with self.trace_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {"batch": self._trace_records}, separators=(",", ":")
                    )
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
