"""Learned start-aligned residual bridge for cross-episode public reanchoring."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .v342_temporal_blended_public_reanchor_runtime import (
    Track2V342TemporalBlendedPublicReanchor,
)


PROFILE_FORMAT = "strict-track2-v350-public-residual-bridge-profile-v1"


class Track2V350LearnedResidualBridge(Track2V342TemporalBlendedPublicReanchor):
    """Align the retrieved motion at frame zero and converge to its terminal state."""

    def __init__(self, *args, bridge_profile=None, **kwargs):
        super().__init__(*args, reanchor_alpha=0.0, **kwargs)
        path = Path(bridge_profile or os.environ.get("WAM_V350_BRIDGE_PROFILE", ""))
        manifest_path = path.with_suffix(".manifest.json")
        if not path.is_file() or not manifest_path.is_file():
            raise RuntimeError(f"v350 residual bridge profile is missing: {path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format") != PROFILE_FORMAT or manifest.get("fit_passed") is not True:
            raise RuntimeError("v350 bridge profile is unsupported or failed")
        if manifest.get("guards", {}).get("public_train_only") is not True:
            raise RuntimeError("v350 bridge profile is not public-train-only")
        with np.load(path, allow_pickle=False) as payload:
            coefficients = payload["bridge_coefficients"].astype(np.float32)
        if coefficients.shape != (5,) or not np.all(np.isfinite(coefficients)):
            raise RuntimeError("invalid v350 bridge coefficients")
        if np.any(coefficients < 0) or np.any(coefficients > 1):
            raise RuntimeError("v350 bridge coefficients outside [0,1]")
        if np.any(np.diff(coefficients) > 1e-6):
            raise RuntimeError("v350 bridge coefficients are not non-increasing")
        self.bridge_profile_path = path
        self.bridge_coefficients = coefficients

    def _apply_reanchor(self, prediction, context, history, future):
        accepted, ood_probability = self._recursive_reanchor_gate(context, history, future)
        if not accepted:
            return prediction, False, ood_probability, None
        row = self._reanchor_row(context, history, future)
        target = self._target(row)
        output = prediction.copy()
        baseline = output[-5:].astype(np.float32)
        retrieval = target[-5:].astype(np.float32)
        offset = baseline[0] - retrieval[0]
        bridge = retrieval + self.bridge_coefficients.reshape(5, 1, 1, 1) * offset
        closed = (future[3:, 13] < 0.5).reshape(5, 1, 1, 1)
        output[-5:] = np.where(
            closed,
            np.clip(np.rint(bridge), 0, 255).astype(np.uint8),
            output[-5:],
        )
        return output, True, ood_probability, row
