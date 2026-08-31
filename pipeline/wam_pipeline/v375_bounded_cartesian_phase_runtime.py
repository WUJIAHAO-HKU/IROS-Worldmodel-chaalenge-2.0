"""Bounded Cartesian phase refinement for sharp, action-causal right-arm video."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import numpy as np
import torch

from .object_geometry_v170 import ActionPoseProjector, denormalize_pose, normalize_action
from .v216_public_knn_blend_runtime import _visual_descriptor
from .v245_clean_progressive_successor_runtime import Track2V245CleanProgressiveSuccessor


MANIFEST_FORMAT = "strict-track2-v375-bounded-cartesian-phase-release-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Track2V375BoundedCartesianPhase(Track2V245CleanProgressiveSuccessor):
    """Refine a visual KNN phase locally using predicted end-effector geometry."""

    def __init__(self, checkpoint_dir, library_index, device="cuda"):
        super().__init__(checkpoint_dir, library_index, device)
        manifest_path = self.root / "cartesian_phase_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format") != MANIFEST_FORMAT:
            raise RuntimeError("unsupported v375 Cartesian phase manifest")
        pose_path = self.root / manifest["pose_checkpoint"]
        if _sha256(pose_path) != manifest["pose_checkpoint_sha256"]:
            raise RuntimeError("v375 pose checkpoint hash mismatch")
        if manifest.get("reward_or_outcomes_used") is not False:
            raise RuntimeError("v375 pose/phase fit must not use reward or outcomes")
        self.radius = int(manifest["radius"])
        self.penalty = float(manifest["penalty"])
        self.pose_scale = np.asarray(manifest["pose_scale"], np.float32)
        if self.pose_scale.shape != (7,) or not np.isfinite(self.pose_scale).all():
            raise RuntimeError("invalid v375 pose scale")

        state = torch.load(pose_path, map_location="cpu", weights_only=False)
        self.pose_device = torch.device(device)
        self.pose_model = ActionPoseProjector(int(state["hidden"])).to(self.pose_device)
        self.pose_model.load_state_dict(state["model"])
        self.pose_model.eval().requires_grad_(False)
        self.pose_action_mean = torch.from_numpy(state["action_mean"]).to(self.pose_device)
        self.pose_action_std = torch.from_numpy(state["action_std"]).to(self.pose_device)
        self.pose_target_mean = torch.from_numpy(state["target_mean"]).to(self.pose_device)
        self.pose_target_std = torch.from_numpy(state["target_std"]).to(self.pose_device)
        self._frame_cache: dict[tuple[int, int], np.ndarray] = {}
        self._frame_lock = threading.Lock()
        self.reference_pose, self.reference_length = self._build_reference_pose()
        self.last_base_index = None
        self.last_selected_frames = None

    @torch.inference_mode()
    def _predict_right_pose(self, actions: np.ndarray) -> np.ndarray:
        value = torch.from_numpy(np.asarray(actions, np.float32)).to(self.pose_device)
        arms = torch.ones(len(value), dtype=torch.long, device=self.pose_device)
        normalized = normalize_action(value, self.pose_action_mean, self.pose_action_std, arms)
        output = self.pose_model(normalized, arms)
        output = denormalize_pose(
            output, self.pose_target_mean, self.pose_target_std, arms
        )
        return output.float().cpu().numpy()

    def _build_reference_pose(self):
        output = {}
        lengths = {}
        for episode, lookup in self.episode_starts.items():
            starts = sorted(lookup)
            frame_count = int(starts[-1] + 13)
            actions = np.full((frame_count - 1, 7), np.nan, np.float32)
            for start in starts:
                with np.load(self.paths[lookup[start]], allow_pickle=False) as values:
                    sequence = np.concatenate(
                        (values["history_actions"], values["future_actions"]), axis=0
                    ).astype(np.float32)[:, 7:14]
                actions[start : start + 12] = sequence
            if not np.isfinite(actions).all():
                raise RuntimeError(f"incomplete v375 action reconstruction for {episode}")
            pose = np.full((frame_count, 7), np.nan, np.float32)
            pose[1:] = self._predict_right_pose(actions)
            output[int(episode)] = pose
            lengths[int(episode)] = frame_count
        return output, lengths

    def _visual_base(self, context: np.ndarray) -> int:
        distance = ((self.visual[self.clean_rows] - _visual_descriptor(context[-1])) ** 2).mean(1)
        return int(self.clean_rows[int(np.argmin(distance))])

    def _frame(self, episode: int, frame_index: int) -> np.ndarray:
        frame_index = int(np.clip(frame_index, 0, self.reference_length[episode] - 1))
        key = (episode, frame_index)
        with self._frame_lock:
            cached = self._frame_cache.get(key)
            if cached is not None:
                return cached
            lookup = self.episode_starts[episode]
            starts = np.asarray(sorted(lookup), np.int64)
            proposed = int(np.clip(frame_index - 5, starts[0], starts[-1]))
            start = int(starts[int(np.argmin(np.abs(starts - proposed)))])
            with np.load(self.paths[lookup[start]], allow_pickle=False) as values:
                if frame_index <= start + 4:
                    frame = values["context_frames"][frame_index - start].copy()
                else:
                    frame = values["target_frames"][frame_index - start - 5].copy()
            self._frame_cache[key] = frame
            return frame

    def _right_prediction(self, parent, context, history, future):
        base = self._visual_base(context)
        episode = int(self.row_episode[base])
        base_start = int(self.row_start[base])
        query_pose = self._predict_right_pose(future[:, 7:14])
        reference = self.reference_pose[episode]
        frame_count = self.reference_length[episode]
        centers = np.clip(np.arange(base_start + 5, base_start + 13), 1, frame_count - 1)
        selected = []
        cursor = 1
        for pose, center in zip(query_pose, centers, strict=True):
            low = max(cursor, int(center) - self.radius, 1)
            high = min(frame_count - 1, int(center) + self.radius)
            rows = np.arange(low, high + 1, dtype=np.int64)
            pose_cost = np.mean(((reference[rows] - pose) / self.pose_scale) ** 2, axis=1)
            offset = (rows - center) / max(self.radius, 1)
            chosen = int(rows[np.argmin(pose_cost + self.penalty * offset ** 2)])
            selected.append(chosen)
            cursor = chosen
        selected = np.asarray(selected, np.int64)
        anchor = self._frame(episode, base_start + 4).astype(np.float32)
        retrieved = np.stack([self._frame(episode, index) for index in selected]).astype(np.float32)
        output = np.clip(
            np.rint(context[-1].astype(np.float32)[None] + retrieved - anchor[None]),
            0,
            255,
        ).astype(np.uint8)
        self.last_base_index = base
        self.last_retrieval_index = base
        self.last_selected_frames = selected.tolist()
        return output

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        parent = self.parent.predict(context_frames, history_actions, future_actions, seed, instruction)
        arm = self.parent.active_arm(history_actions, future_actions, instruction or "")
        self.last_arm_route = arm
        if arm != "right":
            self.last_base_index = None; self.last_selected_frames = None
            return parent
        return self._right_prediction(parent, context_frames, history_actions, future_actions)

    def predict_batch(self, context_frames, history_actions, future_actions, seeds, instructions):
        parent = self.parent.predict_batch(context_frames, history_actions, future_actions, seeds, instructions)
        routes = [
            self.parent.active_arm(history, future, instruction or "")
            for history, future, instruction in zip(history_actions, future_actions, instructions, strict=True)
        ]
        output = parent.copy()
        for index, arm in enumerate(routes):
            if arm == "right":
                output[index] = self._right_prediction(
                    parent[index], context_frames[index], history_actions[index], future_actions[index]
                )
        self.last_arm_route = "mixed" if len(set(routes)) > 1 else routes[0]
        return output
