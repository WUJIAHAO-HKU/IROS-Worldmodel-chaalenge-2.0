"""Threaded 0.90-strength refinement of the v340 public reanchor."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .v334_bounded_onset_terminal_runtime import Track2V334BoundedOnsetTerminal
from .v338_recursive_public_reanchor_runtime import (
    PHASE_MAX_OFFSET,
    Track2V338RecursivePublicReanchor,
)


TEMPORAL_REANCHOR_ALPHA = 0.90
DEFAULT_FEATURE_WORKERS = 8


class Track2V342TemporalBlendedPublicReanchor(Track2V338RecursivePublicReanchor):
    """Use a 0.90 next-context blend and parallelize independent batch gates."""

    def __init__(self, *args, feature_workers=None, **kwargs):
        kwargs.setdefault("reanchor_alpha", TEMPORAL_REANCHOR_ALPHA)
        kwargs.setdefault("phase_max_offset", PHASE_MAX_OFFSET)
        super().__init__(*args, **kwargs)
        requested = int(feature_workers or os.environ.get("WAM_V342_FEATURE_WORKERS", DEFAULT_FEATURE_WORKERS))
        self.feature_workers = max(1, min(requested, DEFAULT_FEATURE_WORKERS))
        self._feature_executor = ThreadPoolExecutor(
            max_workers=self.feature_workers, thread_name_prefix="v342-reanchor"
        )

    def predict_batch(
        self,
        context_frames,
        history_actions,
        future_actions,
        seeds,
        instructions,
    ):
        baseline = Track2V334BoundedOnsetTerminal.predict_batch(
            self, context_frames, history_actions, future_actions, seeds, instructions
        )
        arguments = list(zip(
            baseline, context_frames, history_actions, future_actions, strict=True
        ))

        def apply(argument):
            return self._apply_reanchor(*argument)

        results = list(self._feature_executor.map(apply, arguments))
        outputs = np.stack([result[0] for result in results], axis=0)
        self.last_recursive_reanchor_mask = np.asarray(
            [result[1] for result in results], dtype=bool
        )
        self.last_recursive_ood_probabilities = np.asarray(
            [result[2] for result in results], dtype=np.float64
        )
        self.last_recursive_reanchor_rows = np.asarray(
            [-1 if result[3] is None else result[3] for result in results], dtype=np.int64
        )
        self.last_recursive_reanchor_count = int(self.last_recursive_reanchor_mask.sum())
        self.last_recursive_reanchor = bool(self.last_recursive_reanchor_count)
        self.last_recursive_ood_probability = (
            float(self.last_recursive_ood_probabilities[0]) if len(results) == 1 else None
        )
        self.last_recursive_reanchor_row = (
            int(self.last_recursive_reanchor_rows[0])
            if len(results) == 1 and self.last_recursive_reanchor_rows[0] >= 0
            else None
        )
        return outputs
