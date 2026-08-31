"""Action-causal right terminal mirror with a public-train-only gate.

The gate decides whether a right-arm request is close to a successful public
expert transport.  Expert-like chunks use the full mirrored v271 successor;
other right chunks use only the mirrored parametric dynamics parent, preventing
successful retrieval frames from being attached to physically invalid actions.
Left requests remain the exact v271 path.  Only RGB frames are returned.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .v271_endpoint_calibrated_terminal_runtime import (
    Track2V271EndpointCalibratedTerminal,
)
from .v290_right_closed_mirror_runtime import (
    mirror_actions,
    mirror_prompt,
    route_right,
)


def action_features(history: np.ndarray, future: np.ndarray) -> np.ndarray:
    anchor = history[-1, 7:13]
    relative = future[:, 7:13] - anchor
    sequence = np.concatenate((anchor[None], future[:, 7:13]), axis=0)
    delta = np.diff(sequence, axis=0)
    path = np.linalg.norm(delta, axis=1)
    net = float(np.linalg.norm(relative[-1]))
    summary = np.asarray(
        [
            path.sum(),
            net,
            net / max(float(path.sum()), 1e-8),
            path.mean(),
            path.max(),
            np.linalg.norm(np.diff(delta, axis=0), axis=1).mean(),
        ],
        dtype=np.float32,
    )
    gripper = np.concatenate((history[-1:, 13], future[:, 13])).astype(np.float32)
    return np.concatenate((relative.reshape(-1), delta.reshape(-1), gripper, summary))


class Track2V312CausalTerminalMirror(Track2V271EndpointCalibratedTerminal):
    def __init__(self, checkpoint_dir, library_index, device="cuda", action_gate=None):
        super().__init__(checkpoint_dir, library_index, device)
        path = Path(action_gate or os.environ.get("WAM_V312_ACTION_GATE", ""))
        if not path.is_file():
            raise RuntimeError(f"v312 action gate is missing: {path}")
        with np.load(path, allow_pickle=False) as values:
            version = str(values["feature_version"].item())
            if version != "v311-relative-delta-gripper-summary-v1":
                raise RuntimeError(f"unsupported v312 gate feature version: {version}")
            self.gate_mean = values["feature_mean"].astype(np.float32)
            self.gate_scale = values["feature_scale"].astype(np.float32)
            self.gate_coefficient = values["coefficient"].astype(np.float32)
            self.gate_intercept = float(values["intercept"])
            self.gate_threshold = float(values["threshold"])
        expected = action_features(
            np.zeros((4, 14), dtype=np.float32),
            np.zeros((8, 14), dtype=np.float32),
        ).shape
        if not (
            self.gate_mean.shape
            == self.gate_scale.shape
            == self.gate_coefficient.shape
            == expected
        ):
            raise RuntimeError("v312 action gate dimensions are inconsistent")
        if not np.all(self.gate_scale > 0):
            raise RuntimeError("v312 action gate has nonpositive scale")
        self.action_gate_path = path
        self.last_gate_probability = None
        self.last_gate_accepted = False
        self.last_right_route = False

    def _probability(self, history: np.ndarray, future: np.ndarray) -> float:
        feature = action_features(history, future)
        normalized = (feature - self.gate_mean) / self.gate_scale
        logit = float(normalized @ self.gate_coefficient + self.gate_intercept)
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0))))

    @staticmethod
    def _unmirror(frames: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(frames[:, :, ::-1, :])

    def _mirrored_successor(self, context, history, future, seed, instruction):
        prediction = super().predict(
            np.ascontiguousarray(context[:, :, ::-1, :]),
            mirror_actions(history),
            mirror_actions(future),
            seed,
            mirror_prompt(instruction),
        )
        return self._unmirror(prediction)

    def _mirrored_parametric(self, context, history, future, seed, instruction):
        prediction = self.parent.predict(
            np.ascontiguousarray(context[:, :, ::-1, :]),
            mirror_actions(history),
            mirror_actions(future),
            seed,
            mirror_prompt(instruction),
        )
        return self._unmirror(prediction)

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        self.last_right_route = route_right(history_actions, future_actions)
        if not self.last_right_route:
            self.last_gate_probability = None
            self.last_gate_accepted = False
            return super().predict(
                context_frames, history_actions, future_actions, seed, instruction
            )
        probability = self._probability(history_actions, future_actions)
        self.last_gate_probability = probability
        self.last_gate_accepted = probability >= self.gate_threshold
        if self.last_gate_accepted:
            return self._mirrored_successor(
                context_frames, history_actions, future_actions, seed, instruction
            )
        return self._mirrored_parametric(
            context_frames, history_actions, future_actions, seed, instruction
        )

    def predict_batch(
        self,
        context_frames,
        history_actions,
        future_actions,
        seeds,
        instructions,
    ):
        routes = np.asarray(
            [
                route_right(history, future)
                for history, future in zip(history_actions, future_actions, strict=True)
            ],
            dtype=bool,
        )
        probabilities = np.asarray(
            [
                self._probability(history, future) if right else np.nan
                for history, future, right in zip(
                    history_actions, future_actions, routes, strict=True
                )
            ],
            dtype=np.float32,
        )
        accepted = routes & (probabilities >= self.gate_threshold)
        rejected = routes & ~accepted
        output = np.empty(
            (len(context_frames), 8, 256, 256, 3), dtype=np.uint8
        )
        direct = ~routes
        if direct.any():
            output[direct] = super().predict_batch(
                context_frames[direct],
                history_actions[direct],
                future_actions[direct],
                np.asarray(seeds)[direct],
                [instructions[index] for index in np.flatnonzero(direct)],
            )
        mirrored_context = np.ascontiguousarray(context_frames[:, :, :, ::-1, :])
        mirrored_history = mirror_actions(history_actions)
        mirrored_future = mirror_actions(future_actions)
        if accepted.any():
            selected = np.flatnonzero(accepted)
            prediction = super().predict_batch(
                mirrored_context[accepted],
                mirrored_history[accepted],
                mirrored_future[accepted],
                np.asarray(seeds)[accepted],
                [mirror_prompt(instructions[index]) for index in selected],
            )
            output[accepted] = np.ascontiguousarray(prediction[:, :, :, ::-1, :])
        if rejected.any():
            selected = np.flatnonzero(rejected)
            prediction = self.parent.predict_batch(
                mirrored_context[rejected],
                mirrored_history[rejected],
                mirrored_future[rejected],
                np.asarray(seeds)[rejected],
                [mirror_prompt(instructions[index]) for index in selected],
            )
            output[rejected] = np.ascontiguousarray(prediction[:, :, :, ::-1, :])
        self.last_right_route = bool(routes.any())
        self.last_gate_probability = (
            float(probabilities[routes][0]) if int(routes.sum()) == 1 else None
        )
        self.last_gate_accepted = bool(accepted.any())
        return output
