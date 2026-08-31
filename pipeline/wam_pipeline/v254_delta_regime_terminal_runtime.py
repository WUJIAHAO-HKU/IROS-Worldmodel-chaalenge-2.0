"""Terminal successor gated by offset-invariant delta-action regimes.

Only request RGB/actions/instruction and a frozen public expert library are
used. Rewards, outcomes, request identity, seeds, and evaluation metadata are
never read.
"""
from __future__ import annotations

import numpy as np

from .v216_public_knn_blend_runtime import _visual_descriptor
from .v245_clean_progressive_successor_runtime import ACTION_WEIGHT, MOTION_SCALE, PREGRASP_ALPHA, _path_length
from .v250_specific_terminal_successor_runtime import Track2V250SpecificTerminalSuccessor

CLEAN_RATIO_MAX = 0.01
CLEAN_ALIGNMENT_FLOOR = 0.95
OOD_RATIO_MIN = 0.80
OOD_ALIGNMENT_FLOOR = 0.50
REGIME_ALPHA_SCALE = 8.0


class Track2V254DeltaRegimeTerminal(Track2V250SpecificTerminalSuccessor):
    def __init__(self, checkpoint_dir, library_index, device='cuda'):
        super().__init__(checkpoint_dir, library_index, device)
        raw = self.action[self.clean_rows].reshape(-1, 12, 14) * self.action_std + self.action_mean
        delta = np.diff(raw[:, :, 7:13], axis=1)
        self.delta_scale = np.maximum(delta.reshape(-1, 6).std(axis=0), 1e-6).astype(np.float32)
        self.delta_feature = (delta / self.delta_scale).reshape(len(self.clean_rows), -1).astype(np.float32)

    def _delta_ratio(self, context, history, future) -> float:
        query = np.concatenate((history, future), axis=0)
        query_feature = (np.diff(query[:, 7:13], axis=0) / self.delta_scale).reshape(-1)
        delta_distance = ((self.delta_feature - query_feature) ** 2).mean(axis=1)
        visual_distance = ((self.visual[self.clean_rows] - _visual_descriptor(context[-1])) ** 2).mean(axis=1)
        score = visual_distance / max(float(np.median(visual_distance)), 1e-9)
        score += ACTION_WEIGHT * delta_distance / max(float(np.median(delta_distance)), 1e-9)
        local = int(np.argmin(score))
        return float(delta_distance[local] / max(float(np.median(delta_distance)), 1e-9))

    @staticmethod
    def _regime_alpha(delta_ratio: float, alignment: float, motion: float) -> float:
        if delta_ratio <= CLEAN_RATIO_MAX:
            floor = CLEAN_ALIGNMENT_FLOOR
        elif delta_ratio >= OOD_RATIO_MIN:
            floor = OOD_ALIGNMENT_FLOOR
        else:
            return 0.0
        direction = float(np.clip((alignment - floor) / (1.0 - floor), 0.0, 1.0))
        motion_gate = float(np.clip(motion / MOTION_SCALE, 0.0, 1.0))
        return float(np.clip(REGIME_ALPHA_SCALE * direction * motion_gate, 0.0, 1.0))

    def _alpha_with_context(self, context, history, future, base) -> float:
        raw = self.action[base].reshape(-1, 14) * self.action_std + self.action_mean
        query_delta = future[-1, 7:13] - history[-1, 7:13]
        library_delta = raw[-1, 7:13] - raw[-9, 7:13]
        denominator = float(np.linalg.norm(query_delta) * np.linalg.norm(library_delta))
        alignment = float(np.dot(query_delta, library_delta) / denominator) if denominator > 1e-8 else 0.0
        self.last_delta_ratio = self._delta_ratio(context, history, future)
        self.last_alignment = alignment
        return self._regime_alpha(self.last_delta_ratio, alignment, _path_length(history, future))

    def _right_prediction(self, parent, context, history, future):
        base, _ = self._nearest_clean(context, history, future)
        self.last_base_index = base
        if not self._post_grasp(history, future):
            self.last_delta_ratio = None
            self.last_alignment = None
            self.last_progressive_index = None
            self.last_progressive_alpha = 0.0
            self.last_retrieval_index = base
            return self._blend(parent, self._target(base), future, PREGRASP_ALPHA)
        target = self._terminal_for_context(context)
        alpha = self._alpha_with_context(context, history, future, base)
        self.last_progressive_index = target
        self.last_progressive_alpha = alpha
        self.last_retrieval_index = target
        return self._blend(parent, self._target(target), future, alpha)
