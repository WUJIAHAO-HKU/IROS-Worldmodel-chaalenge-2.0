"""Public-train learned five-frame residual dynamics for recursive reanchoring."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .v342_temporal_blended_public_reanchor_runtime import (
    Track2V342TemporalBlendedPublicReanchor,
)


PROFILE_FORMAT = "strict-track2-v346-public-temporal-residual-profile-v1"


class Track2V346LearnedTemporalResidual(Track2V342TemporalBlendedPublicReanchor):
    """Blend the public residual with a frozen learned monotone horizon profile."""

    def __init__(self, *args, temporal_profile=None, **kwargs):
        super().__init__(*args, reanchor_alpha=0.0, **kwargs)
        path = Path(temporal_profile or os.environ.get("WAM_V346_TEMPORAL_PROFILE", ""))
        manifest_path = path.with_suffix(".manifest.json")
        if not path.is_file() or not manifest_path.is_file():
            raise RuntimeError(f"v346 learned temporal profile is missing: {path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format") != PROFILE_FORMAT:
            raise RuntimeError("unsupported v346 temporal profile")
        if manifest.get("guards", {}).get("public_train_only") is not True:
            raise RuntimeError("v346 profile is not declared public-train-only")
        if manifest.get("fit_passed") is not True:
            raise RuntimeError("v346 temporal profile did not pass its frozen fit gate")
        with np.load(path, allow_pickle=False) as payload:
            coefficients = payload["coefficients"].astype(np.float32)
        if coefficients.shape != (5,) or not np.all(np.isfinite(coefficients)):
            raise RuntimeError("invalid v346 temporal coefficients")
        if np.any(coefficients < 0) or np.any(coefficients > 1):
            raise RuntimeError("v346 temporal coefficients outside [0,1]")
        if np.any(np.diff(coefficients) < -1e-6):
            raise RuntimeError("v346 temporal coefficients are not monotone")
        self.temporal_profile_path = path
        self.temporal_coefficients = coefficients

    def _apply_reanchor(self, prediction, context, history, future):
        accepted, ood_probability = self._recursive_reanchor_gate(context, history, future)
        if not accepted:
            return prediction, False, ood_probability, None
        row = self._reanchor_row(context, history, future)
        target = self._target(row)
        output = prediction.copy()
        closed = (future[3:, 13] < 0.5).astype(np.float32)
        per_frame = (closed * self.temporal_coefficients).reshape(5, 1, 1, 1)
        output[-5:] = np.clip(
            np.rint(
                (1.0 - per_frame) * output[-5:].astype(np.float32)
                + per_frame * target[-5:].astype(np.float32)
            ),
            0, 255,
        ).astype(np.uint8)
        return output, True, ood_probability, row
