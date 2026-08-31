"""Action-causal public-trajectory reanchor for corrupted recursive contexts.

V334 remains the exact default.  A frozen public-train image-quality gate may
identify a recursively generated five-frame context as corrupted.  Only for a
right/post-grasp, high-confidence, non-failure action in a success-phase action
regime, frames 3..7 are reanchored to an action/visual matched sustained-success
trajectory from the public training library.  The runtime reads no reward,
outcome, seed identity, evaluation metadata, or simulator state.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .v216_public_knn_blend_runtime import _visual_descriptor
from .v245_clean_progressive_successor_runtime import ACTION_WEIGHT
from .v290_right_closed_mirror_runtime import route_right
from .v324_phase_guarded_terminal_runtime import ACTION_PROBABILITY_MIN
from .v334_bounded_onset_terminal_runtime import Track2V334BoundedOnsetTerminal
from .v337_public_recursive_ood_gate import PublicRecursiveOODGate


REANCHOR_ALPHA = 1.0
PHASE_MAX_OFFSET = 16


class Track2V338RecursivePublicReanchor(Track2V334BoundedOnsetTerminal):
    """Apply a coherent public next-context only after causal corruption gating."""

    def __init__(
        self,
        checkpoint_dir,
        library_index,
        device="cuda",
        action_gate=None,
        phase_gate=None,
        recursive_ood_gate=None,
        reanchor_alpha=REANCHOR_ALPHA,
        phase_max_offset=PHASE_MAX_OFFSET,
    ):
        super().__init__(checkpoint_dir, library_index, device, action_gate, phase_gate)
        path = Path(recursive_ood_gate or os.environ.get("WAM_V338_RECURSIVE_OOD_GATE", ""))
        if not path.is_file():
            raise RuntimeError(f"v338 recursive OOD gate is missing: {path}")
        self.recursive_ood_gate_path = path
        self.recursive_ood_gate = PublicRecursiveOODGate(path)
        self.reanchor_alpha = float(reanchor_alpha)
        self.phase_max_offset = int(phase_max_offset)
        if not 0.0 <= self.reanchor_alpha <= 1.0:
            raise ValueError("reanchor alpha must be in [0,1]")
        if self.phase_max_offset < 0:
            raise ValueError("phase max offset must be nonnegative")
        rows = []
        for row in self.clean_rows:
            episode = int(self.row_episode[row])
            start = int(self.row_start[row])
            onset = int(self.phase_onset[episode])
            if (
                self.eligible_episode.get(episode, False)
                and onset <= start <= onset + self.phase_max_offset
            ):
                rows.append(int(row))
        self.reanchor_rows = np.asarray(rows, dtype=np.int64)
        if len(self.reanchor_rows) < 32:
            raise RuntimeError(f"too few public phase reanchor rows: {len(self.reanchor_rows)}")
        self.last_recursive_ood_probability = None
        self.last_recursive_reanchor = False
        self.last_recursive_reanchor_row = None

    def _action_base(self, history: np.ndarray, future: np.ndarray) -> int:
        actions = np.concatenate((history, future), axis=0).astype(np.float32)
        query = ((actions - self.action_mean) / self.action_std).reshape(-1)
        distance = ((self.action[self.clean_rows] - query) ** 2).mean(axis=1)
        return int(self.clean_rows[int(np.argmin(distance))])

    def _reanchor_row(self, context: np.ndarray, history: np.ndarray, future: np.ndarray) -> int:
        query_visual = _visual_descriptor(context[-1])
        actions = np.concatenate((history, future), axis=0).astype(np.float32)
        query_action = ((actions - self.action_mean) / self.action_std).reshape(-1)
        rows = self.reanchor_rows
        visual_distance = ((self.visual[rows] - query_visual) ** 2).mean(axis=1)
        action_distance = ((self.action[rows] - query_action) ** 2).mean(axis=1)
        score = visual_distance / max(float(np.median(visual_distance)), 1e-9)
        score += ACTION_WEIGHT * action_distance / max(float(np.median(action_distance)), 1e-9)
        return int(rows[int(np.argmin(score))])

    def _recursive_reanchor_gate(
        self,
        context: np.ndarray,
        history: np.ndarray,
        future: np.ndarray,
    ) -> tuple[bool, float]:
        probability = float(self._probability(history, future))
        failure = self._signature(history, future, probability)
        ood_probability = self.recursive_ood_gate.probability(context)
        action_base = self._action_base(history, future)
        action_phase_ready, _, _, _ = self._phase(action_base)
        accepted = bool(
            route_right(history, future)
            and self._post_grasp(history, future)
            and probability >= ACTION_PROBABILITY_MIN
            and failure is None
            and action_phase_ready
            and ood_probability >= self.recursive_ood_gate.threshold
        )
        return accepted, ood_probability

    def _apply_reanchor(
        self,
        prediction: np.ndarray,
        context: np.ndarray,
        history: np.ndarray,
        future: np.ndarray,
    ) -> tuple[np.ndarray, bool, float, int | None]:
        accepted, ood_probability = self._recursive_reanchor_gate(context, history, future)
        if not accepted:
            return prediction, False, ood_probability, None
        row = self._reanchor_row(context, history, future)
        target = self._target(row)
        output = prediction.copy()
        per_frame = (
            (future[3:, 13] < 0.5).astype(np.float32) * self.reanchor_alpha
        ).reshape(5, 1, 1, 1)
        output[-5:] = np.clip(
            np.rint(
                (1.0 - per_frame) * output[-5:].astype(np.float32)
                + per_frame * target[-5:].astype(np.float32)
            ),
            0,
            255,
        ).astype(np.uint8)
        return output, True, ood_probability, row

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        baseline = super().predict(
            context_frames, history_actions, future_actions, seed, instruction
        )
        output, accepted, probability, row = self._apply_reanchor(
            baseline, context_frames, history_actions, future_actions
        )
        self.last_recursive_ood_probability = probability
        self.last_recursive_reanchor = accepted
        self.last_recursive_reanchor_row = row
        return output

    def predict_batch(
        self,
        context_frames,
        history_actions,
        future_actions,
        seeds,
        instructions,
    ):
        baseline = super().predict_batch(
            context_frames, history_actions, future_actions, seeds, instructions
        )
        outputs = []
        metadata = []
        for prediction, context, history, future in zip(
            baseline, context_frames, history_actions, future_actions, strict=True
        ):
            output, accepted, probability, row = self._apply_reanchor(
                prediction, context, history, future
            )
            outputs.append(output)
            metadata.append((accepted, probability, row))
        self.last_recursive_reanchor_count = int(sum(value[0] for value in metadata))
        self.last_recursive_reanchor = bool(self.last_recursive_reanchor_count)
        self.last_recursive_ood_probability = (
            metadata[0][1] if len(metadata) == 1 else None
        )
        self.last_recursive_reanchor_row = metadata[0][2] if len(metadata) == 1 else None
        return np.stack(outputs, axis=0)
