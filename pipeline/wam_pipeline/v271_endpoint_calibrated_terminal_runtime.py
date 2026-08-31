"""Delta-regime terminal successor with endpoint-calibrated OOD rewards.

The runtime uses only request RGB/actions/instruction and a frozen public
expert library. It never reads rewards, outcomes, seeds, request identity, or
evaluation metadata, and it returns only RGB frames.
"""
from __future__ import annotations

import numpy as np

from .v254_delta_regime_terminal_runtime import OOD_RATIO_MIN, Track2V254DeltaRegimeTerminal

ENDPOINT_TEMPERATURE = 0.25
PROGRESS_FLOOR = -0.25
PROGRESS_CEIL = 0.50
OOD_ALPHA_BOOST = 2.0


class Track2V271EndpointCalibratedTerminal(Track2V254DeltaRegimeTerminal):
    """Keep v254 clean/mid gates and calibrate only its OOD terminal blend."""

    @staticmethod
    def _endpoint_quality(endpoint_distance: float, relative_progress: float) -> float:
        endpoint_confidence = float(np.exp(-max(endpoint_distance, 0.0) / ENDPOINT_TEMPERATURE))
        progress_confidence = float(
            np.clip(
                (relative_progress - PROGRESS_FLOOR) / (PROGRESS_CEIL - PROGRESS_FLOOR),
                0.0,
                1.0,
            )
        )
        return float(np.sqrt(endpoint_confidence * progress_confidence))

    def _absolute_endpoint_metrics(self, history, future, base) -> tuple[float, float]:
        raw = self.action[base].reshape(-1, 14) * self.action_std + self.action_mean
        scale = np.maximum(self.action_std[7:13], 1e-6)
        start_distance = float(np.mean(((history[-1, 7:13] - raw[-9, 7:13]) / scale) ** 2))
        endpoint_distance = float(np.mean(((future[-1, 7:13] - raw[-1, 7:13]) / scale) ** 2))
        relative_progress = float(
            (start_distance - endpoint_distance) / max(start_distance, 1e-9)
        )
        return endpoint_distance, relative_progress

    def _alpha_with_context(self, context, history, future, base) -> float:
        base_alpha = super()._alpha_with_context(context, history, future, base)
        endpoint_distance, relative_progress = self._absolute_endpoint_metrics(
            history, future, base
        )
        self.last_endpoint_distance = endpoint_distance
        self.last_relative_endpoint_progress = relative_progress
        self.last_endpoint_quality = self._endpoint_quality(
            endpoint_distance, relative_progress
        )
        if self.last_delta_ratio is not None and self.last_delta_ratio >= OOD_RATIO_MIN:
            return float(
                np.clip(
                    OOD_ALPHA_BOOST * base_alpha * self.last_endpoint_quality,
                    0.0,
                    1.0,
                )
            )
        return base_alpha

