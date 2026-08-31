"""Public-training-only right-arm kNN blended with a frozen parametric parent.

The runtime obeys the Track-2 contract: request RGB/actions/instruction are its
only query inputs and its only output is eight future RGB frames.  Retrieval
never reads outcome, reward, request id, seed, or evaluation metadata.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import numpy as np

from .arm_routed_autoregressive_runtime import Track2ArmRoutedAutoregressiveUNet


ALPHA = 0.70
VISUAL_WEIGHT = 1.0
ACTION_WEIGHT = 2.5
LIBRARY_FORMAT = "strict-track2-v214-public-right-knn-library-v1"
PARENT_FORMAT = "track2-arm-routed-autoregressive-release-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _visual_descriptor(frame: np.ndarray) -> np.ndarray:
    value = (
        frame.astype(np.float32)
        .reshape(16, 16, 16, 16, 3)
        .mean((1, 3))
        / 255.0
    )
    return value.reshape(-1)


class Track2V216PublicKNNBlend:
    """Frozen v209 parent plus fixed public-data kNN RGB blend."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        library_index: str | Path,
        device: str = "cuda",
    ) -> None:
        self.root = Path(checkpoint_dir)
        self.library_index = Path(library_index)
        parent_manifest = json.loads(
            (self.root / "arm_routed_autoregressive_manifest.json").read_text()
        )
        if parent_manifest.get("format") != PARENT_FORMAT:
            raise RuntimeError("unsupported v216 parent release")
        for relative, expected in parent_manifest["model_sha256"].items():
            path = self.root / relative / "model.pt"
            if not path.is_file() or _sha256(path) != expected:
                raise RuntimeError(f"v216 parent expert hash mismatch: {path}")
        library_manifest_path = self.library_index.with_suffix(".manifest.json")
        library_manifest = json.loads(library_manifest_path.read_text())
        if library_manifest.get("format") != LIBRARY_FORMAT:
            raise RuntimeError("unsupported v216 public retrieval library")
        if library_manifest.get("data_policy") != "declared public training episodes only":
            raise RuntimeError("v216 retrieval library is not public-training-only")
        if library_manifest.get("outcome_labels_used_for_index_or_retrieval") is not False:
            raise RuntimeError("v216 retrieval library must not use outcomes")
        if library_manifest.get("hidden_or_final_data") is not False:
            raise RuntimeError("v216 retrieval library contains prohibited data")

        with np.load(self.library_index, allow_pickle=False) as values:
            self.paths = values["path"].astype(str)
            self.visual = values["visual"].astype(np.float32)
            self.action = values["action"].astype(np.float32)
            self.action_mean = values["normalization_mean"].astype(np.float32)
            self.action_std = values["normalization_std"].astype(np.float32)
        if len(self.paths) == 0 or self.visual.shape[0] != self.action.shape[0]:
            raise RuntimeError("v216 retrieval index is empty or inconsistent")

        self.parent = Track2ArmRoutedAutoregressiveUNet(
            self.root / parent_manifest["left_expert"],
            self.root / parent_manifest["right_expert"],
            device,
        )
        self._target_cache: dict[int, np.ndarray] = {}
        self._target_lock = threading.Lock()
        self.last_arm_route: str | None = None
        self.last_retrieval_index: int | None = None

    def _nearest(self, context: np.ndarray, history: np.ndarray, future: np.ndarray) -> int:
        query_visual = _visual_descriptor(context[-1])
        actions = np.concatenate((history, future), axis=0).astype(np.float32)
        query_action = ((actions - self.action_mean) / self.action_std).reshape(-1)
        visual_distance = ((self.visual - query_visual) ** 2).mean(1)
        action_distance = ((self.action - query_action) ** 2).mean(1)
        visual_scale = max(float(np.median(visual_distance)), 1e-9)
        action_scale = max(float(np.median(action_distance)), 1e-9)
        score = VISUAL_WEIGHT * visual_distance / visual_scale
        score += ACTION_WEIGHT * action_distance / action_scale
        return int(np.argmin(score))

    def _target(self, index: int) -> np.ndarray:
        with self._target_lock:
            target = self._target_cache.get(index)
            if target is None:
                with np.load(self.paths[index], allow_pickle=False) as values:
                    target = values["target_frames"].copy()
                if target.shape != (8, 256, 256, 3) or target.dtype != np.uint8:
                    raise RuntimeError(f"invalid public retrieval target: {self.paths[index]}")
                self._target_cache[index] = target
            return target

    @staticmethod
    def _blend(parent: np.ndarray, retrieval: np.ndarray) -> np.ndarray:
        return np.clip(
            np.rint(
                (1.0 - ALPHA) * parent.astype(np.float32)
                + ALPHA * retrieval.astype(np.float32)
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
        return self._blend(parent, self._target(selected))

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
            output[index] = self._blend(parent[index], self._target(selected))
            selected_rows.append(selected)
        self.last_arm_route = "mixed" if len(set(routes)) > 1 else routes[0]
        self.last_retrieval_index = (
            selected_rows[0] if len(selected_rows) == 1 else None
        )
        return output
