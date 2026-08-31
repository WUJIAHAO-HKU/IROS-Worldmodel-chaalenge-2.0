"""Causal right-gripper gate for the public-training retrieval world model."""

from __future__ import annotations

import numpy as np

from .v216_public_knn_blend_runtime import ALPHA, Track2V216PublicKNNBlend


class Track2V236CausalPublicKNNBlend(Track2V216PublicKNNBlend):
    """Apply public retrieval only while the predicted right gripper is closed."""

    @staticmethod
    def causal_blend(
        parent: np.ndarray,
        retrieval: np.ndarray,
        future_actions: np.ndarray,
    ) -> np.ndarray:
        if parent.shape != (8, 256, 256, 3) or retrieval.shape != parent.shape:
            raise ValueError("v236 expects two [8,256,256,3] prediction chunks")
        actions = np.asarray(future_actions, dtype=np.float32)
        if actions.shape != (8, 14):
            raise ValueError(f"v236 expects [8,14] future actions, got {actions.shape}")
        alpha = (actions[:, 13] < 0.5).astype(np.float32) * ALPHA
        alpha = alpha.reshape(8, 1, 1, 1)
        return np.clip(
            np.rint(
                (1.0 - alpha) * parent.astype(np.float32)
                + alpha * retrieval.astype(np.float32)
            ),
            0,
            255,
        ).astype(np.uint8)

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        parent = self.parent.predict(
            context_frames, history_actions, future_actions, seed, instruction
        )
        arm = self.parent.active_arm(history_actions, future_actions, instruction or "")
        self.last_arm_route = arm
        if arm != "right":
            self.last_retrieval_index = None
            return parent
        selected = self._nearest(context_frames, history_actions, future_actions)
        self.last_retrieval_index = selected
        return self.causal_blend(parent, self._target(selected), future_actions)

    def predict_batch(
        self,
        context_frames: np.ndarray,
        history_actions: np.ndarray,
        future_actions: np.ndarray,
        seeds: np.ndarray,
        instructions: list[str],
    ) -> np.ndarray:
        parent = self.parent.predict_batch(
            context_frames, history_actions, future_actions, seeds, instructions
        )
        routes = [
            self.parent.active_arm(history, future, instruction or "")
            for history, future, instruction in zip(
                history_actions, future_actions, instructions
            )
        ]
        output = parent.copy()
        selected_rows = []
        for index, arm in enumerate(routes):
            if arm != "right":
                selected_rows.append(None)
                continue
            selected = self._nearest(
                context_frames[index], history_actions[index], future_actions[index]
            )
            output[index] = self.causal_blend(
                parent[index], self._target(selected), future_actions[index]
            )
            selected_rows.append(selected)
        self.last_arm_route = "mixed" if len(set(routes)) > 1 else routes[0]
        self.last_retrieval_index = (
            selected_rows[0] if len(selected_rows) == 1 else None
        )
        return output
