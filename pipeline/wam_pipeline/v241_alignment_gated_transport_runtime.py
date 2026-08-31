"""Public-only phase-aware right-arm transport retrieval world model."""

from __future__ import annotations

import numpy as np

from .v216_public_knn_blend_runtime import Track2V216PublicKNNBlend, _visual_descriptor


BASE_ALPHA = 0.70
TRANSPORT_PATH_QUANTILE = 0.90
TRANSPORT_ACTION_WEIGHT = 4.0
DISTANCE_TEMPERATURE = 10.0
PUBLIC_CAPTURE_DISTANCE_SCALE = 26.628942489624023
ALIGNMENT_FLOOR = 0.50
MOTION_QUANTILE = 0.75


def _transport_descriptor(history: np.ndarray, future: np.ndarray) -> np.ndarray:
    relative = future[..., 7:13] - history[..., -1:, 7:13]
    return relative.reshape(relative.shape[:-2] + (48,))


def _path_length(history: np.ndarray, future: np.ndarray) -> np.ndarray:
    path = np.concatenate((history[..., -1:, 7:13], future[..., 7:13]), axis=-2)
    return np.linalg.norm(np.diff(path, axis=-2), axis=-1).sum(-1)


class Track2V241AlignmentGatedTransport(Track2V216PublicKNNBlend):
    """Use strong public transport windows only for aligned post-grasp actions."""

    def __init__(self, checkpoint_dir, library_index, device="cuda") -> None:
        super().__init__(checkpoint_dir, library_index, device)
        steps = self.action.shape[1] // 14
        raw = self.action.reshape(-1, steps, 14) * self.action_std + self.action_mean
        self.library_history = raw[:, :-8]
        self.library_future = raw[:, -8:]
        closed = (self.library_future[..., 13] < 0.5).mean(-1)
        paths = _path_length(self.library_history, self.library_future)
        threshold = float(np.quantile(paths, TRANSPORT_PATH_QUANTILE))
        self.transport_rows = np.flatnonzero((closed >= 0.75) & (paths >= threshold))
        if self.transport_rows.size != 188:
            raise RuntimeError(
                f"v241 expected 188 frozen public transport rows, got {self.transport_rows.size}"
            )
        descriptors = _transport_descriptor(self.library_history, self.library_future)
        self.transport_mean = descriptors[self.transport_rows].mean(0)
        self.transport_std = descriptors[self.transport_rows].std(0).clip(1e-4)
        self.transport_z = (
            descriptors[self.transport_rows] - self.transport_mean
        ) / self.transport_std
        self.motion_scale = float(np.quantile(paths[self.transport_rows], MOTION_QUANTILE))
        self.last_transport_index: int | None = None
        self.last_transport_alpha: float = 0.0

    @staticmethod
    def _post_grasp(history: np.ndarray, future: np.ndarray) -> bool:
        return bool(history[-1, 13] < 0.5 and (future[:, 13] < 0.5).mean() >= 0.75)

    def _select_transport(
        self, context: np.ndarray, history: np.ndarray, future: np.ndarray
    ) -> tuple[int, float]:
        query = (
            _transport_descriptor(history[None], future[None])[0] - self.transport_mean
        ) / self.transport_std
        action_distance = ((self.transport_z - query) ** 2).mean(1)
        visual_distance = (
            (self.visual[self.transport_rows] - _visual_descriptor(context[-1])) ** 2
        ).mean(1)
        score = visual_distance / max(float(np.median(visual_distance)), 1e-9)
        score += TRANSPORT_ACTION_WEIGHT * action_distance / max(
            float(np.median(action_distance)), 1e-9
        )
        local = int(np.argmin(score))
        return int(self.transport_rows[local]), float(action_distance[local])

    def _transport_alpha(
        self, history: np.ndarray, future: np.ndarray, selected: int, distance: float
    ) -> float:
        query_terminal = future[-1, 7:13] - history[-1, 7:13]
        library_terminal = (
            self.library_future[selected, -1, 7:13]
            - self.library_history[selected, -1, 7:13]
        )
        denominator = float(
            np.linalg.norm(query_terminal) * np.linalg.norm(library_terminal)
        )
        alignment = (
            float(np.dot(query_terminal, library_terminal)) / denominator
            if denominator > 1e-8
            else 0.0
        )
        alignment_gate = float(
            np.clip((alignment - ALIGNMENT_FLOOR) / (1.0 - ALIGNMENT_FLOOR), 0.0, 1.0)
        )
        motion_gate = float(
            np.clip(_path_length(history[None], future[None])[0] / self.motion_scale, 0.0, 1.0)
        )
        confidence = float(
            np.exp(
                -distance
                / (DISTANCE_TEMPERATURE * PUBLIC_CAPTURE_DISTANCE_SCALE)
            )
        )
        return confidence * alignment_gate * motion_gate

    @staticmethod
    def _blend_with_alpha(
        parent: np.ndarray, target: np.ndarray, future: np.ndarray, alpha: float
    ) -> np.ndarray:
        per_frame = (future[:, 13] < 0.5).astype(np.float32) * float(alpha)
        per_frame = per_frame.reshape(8, 1, 1, 1)
        return np.clip(
            np.rint(
                (1.0 - per_frame) * parent.astype(np.float32)
                + per_frame * target.astype(np.float32)
            ),
            0,
            255,
        ).astype(np.uint8)

    def _right_prediction(
        self, parent: np.ndarray, context: np.ndarray, history: np.ndarray, future: np.ndarray
    ) -> np.ndarray:
        if not self._post_grasp(history, future):
            selected = self._nearest(context, history, future)
            self.last_retrieval_index = selected
            self.last_transport_index = None
            self.last_transport_alpha = 0.0
            return self._blend_with_alpha(
                parent, self._target(selected), future, BASE_ALPHA
            )
        selected, distance = self._select_transport(context, history, future)
        alpha = self._transport_alpha(history, future, selected, distance)
        self.last_retrieval_index = selected
        self.last_transport_index = selected
        self.last_transport_alpha = alpha
        return self._blend_with_alpha(parent, self._target(selected), future, alpha)

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        parent = self.parent.predict(
            context_frames, history_actions, future_actions, seed, instruction
        )
        arm = self.parent.active_arm(history_actions, future_actions, instruction or "")
        self.last_arm_route = arm
        if arm != "right":
            self.last_retrieval_index = None
            self.last_transport_index = None
            self.last_transport_alpha = 0.0
            return parent
        return self._right_prediction(parent, context_frames, history_actions, future_actions)

    def predict_batch(self, context_frames, history_actions, future_actions, seeds, instructions):
        parent = self.parent.predict_batch(
            context_frames, history_actions, future_actions, seeds, instructions
        )
        routes = [
            self.parent.active_arm(history, future, instruction or "")
            for history, future, instruction in zip(history_actions, future_actions, instructions)
        ]
        output = parent.copy()
        for index, arm in enumerate(routes):
            if arm == "right":
                output[index] = self._right_prediction(
                    parent[index], context_frames[index], history_actions[index], future_actions[index]
                )
        self.last_arm_route = "mixed" if len(set(routes)) > 1 else routes[0]
        return output
