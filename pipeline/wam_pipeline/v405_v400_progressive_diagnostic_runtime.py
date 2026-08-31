"""Output-equivalent v400 telemetry for reachable progressive successors."""

from __future__ import annotations

import numpy as np

from .v404_v400_dense_diagnostic_runtime import Track2V404V400DenseDiagnostic


class Track2V405V400ProgressiveDiagnostic(Track2V404V400DenseDiagnostic):
    """Add public-expert progressive matching metrics without changing RGB."""

    def _route(self, context, history, future, arm):
        result = super()._route(context, history, future, arm)
        record = self._trace_records[-1]
        record.update(
            {
                "progressive_base_row": None,
                "progressive_target_row": None,
                "progressive_action_distance": None,
                "progressive_alpha": None,
                "progressive_alignment": None,
                "progressive_path_length": None,
                "progressive_target_advance": None,
            }
        )
        if not record["broad_eligible"]:
            return result

        base, distance = self._nearest_clean(context, history, future)
        target = self._progressive_row(base)
        alpha = float(self._alpha(history, future, base, distance))
        raw = self.action[base].reshape(-1, 14) * self.action_std + self.action_mean
        library_history, library_future = raw[:-8], raw[-8:]
        query_terminal = future[-1, 7:13] - history[-1, 7:13]
        library_terminal = (
            library_future[-1, 7:13] - library_history[-1, 7:13]
        )
        denominator = float(
            np.linalg.norm(query_terminal) * np.linalg.norm(library_terminal)
        )
        alignment = (
            float(np.dot(query_terminal, library_terminal) / denominator)
            if denominator > 1e-8
            else 0.0
        )
        sequence = np.concatenate((history[-1:, 7:13], future[:, 7:13]), axis=0)
        path_length = float(np.linalg.norm(np.diff(sequence, axis=0), axis=1).sum())
        record.update(
            {
                "progressive_base_row": int(base),
                "progressive_target_row": int(target),
                "progressive_action_distance": float(distance),
                "progressive_alpha": alpha,
                "progressive_alignment": alignment,
                "progressive_path_length": path_length,
                "progressive_target_advance": int(
                    self.row_start[target] - self.row_start[base]
                ),
            }
        )
        return result
