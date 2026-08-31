"""V440: train-only LOEO-aligned teacher residual on the original v169 RGB."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .autoregressive_unet_runtime import Track2AutoregressiveUNet
from .v169_arm_routed_runtime import Track2V169ArmRoutedRuntime


FORMAT = "track2-v440-v169-trainonly-aligned-projection-release-v1"
INDEX_FORMAT = "strict-track2-v440-trainonly-rgb-alignment-index-v1"
V436_FORMAT = "track2-v436-v432-step25-diagnostic-release-v1"
MU = np.asarray([-0.330, -1.425, -1.563, 1.617, 0.494, 0.779], dtype=np.float64)
PHASES = ("close", "postclose")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _joint_path(actions: np.ndarray, begin: int) -> float:
    return float(np.linalg.norm(np.diff(actions[:, begin : begin + 6], axis=0), axis=1).sum())


def _explicit_right(instruction: str | None) -> bool:
    text = str(instruction or "").lower()
    return "right arm" in text and "left arm" not in text


def _gripper_phase(history: np.ndarray, future: np.ndarray) -> str | None:
    future_closed = future[:, 13] < 0.5
    if history[-1, 13] < 0.5:
        return "postclose" if np.all(future_closed) else None
    close_indices = np.flatnonzero(future_closed)
    if close_indices.size and np.all(future_closed[close_indices[0] :]):
        return "close"
    return None


def gate_decision(history_actions: np.ndarray, future_actions: np.ndarray, instruction: str | None) -> dict:
    history = np.asarray(history_actions, dtype=np.float32)
    future = np.asarray(future_actions, dtype=np.float32)
    if history.ndim != 2 or future.shape != (8, 14) or history.shape[1] != 14:
        raise ValueError(f"v440 expects history[T,14] and future[8,14], got {history.shape}/{future.shape}")
    actions = np.concatenate((history, future), axis=0)
    left_path = _joint_path(actions, 0)
    right_path = _joint_path(actions, 7)
    phase = _gripper_phase(history, future)
    projection = float(np.dot(future[-1, 7:13] - history[-1, 7:13], MU))
    explicit = _explicit_right(instruction)
    gate = bool(explicit and right_path > left_path and phase in PHASES and projection > 0.0)
    if not explicit:
        reason = "prompt_not_explicit_right"
    elif right_path <= left_path:
        reason = "right_path_not_greater_than_left"
    elif phase not in PHASES:
        reason = "right_gripper_not_close_or_held"
    elif projection <= 0.0:
        reason = "right_delta_projection_not_positive"
    else:
        reason = f"right_{phase}_aligned_projection"
    return {
        "gate": gate, "phase": phase, "explicit_right": explicit,
        "right_path": right_path, "left_path": left_path,
        "delta_dot_mu": projection, "reason": reason,
    }


def clipped_rgb_delta(candidate: np.ndarray, parent: np.ndarray) -> np.ndarray:
    candidate = np.asarray(candidate)
    parent = np.asarray(parent)
    if candidate.shape != parent.shape or candidate.shape[-1] != 3:
        raise ValueError("v440 teacher RGB shapes do not match")
    return np.clip(candidate.astype(np.int16) - parent.astype(np.int16), -8, 8).astype(np.int16)


def apply_aligned_projection(baseline: np.ndarray, candidate: np.ndarray, parent: np.ndarray, beta: np.ndarray | None) -> np.ndarray:
    baseline = np.asarray(baseline)
    if baseline.dtype != np.uint8 or baseline.ndim != 4 or baseline.shape[0] != 8 or baseline.shape[-1] != 3:
        raise ValueError(f"v440 expects baseline uint8 RGB[8,H,W,3], got {baseline.shape}/{baseline.dtype}")
    if beta is None:
        return baseline.copy()
    coefficient = np.asarray(beta, dtype=np.float64)
    if coefficient.shape != (8, 3) or not np.isfinite(coefficient).all() or np.abs(coefficient).max() > 1.0:
        raise ValueError("v440 beta must be finite [8,3] and bounded by one")
    delta = clipped_rgb_delta(candidate, parent).astype(np.float64)
    output = np.rint(baseline.astype(np.float64) + coefficient[:, None, None, :] * delta)
    return np.clip(output, 0, 255).astype(np.uint8)


class Track2V440V169TrainOnlyAlignedProjection:
    def __init__(self, release_dir: str | Path, device: str = "cuda") -> None:
        root = Path(release_dir).resolve()
        manifest_path = root / "v440_aligned_projection_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format") != FORMAT or manifest.get("official_reward_runtime_used") is not False:
            raise RuntimeError("unsupported or reward-coupled v440 release")
        paths = {
            "alignment_index": root / manifest["alignment_index"],
            "v169_manifest": root / manifest["v169_release"] / "v169_arm_routed_manifest.json",
            "v436_manifest": root / manifest["v436_release"] / "v436_diagnostic_manifest.json",
        }
        for key, path in paths.items():
            if not path.is_file() or _sha256(path) != manifest["sha256"][key]:
                raise RuntimeError(f"v440 release hash mismatch: {key}")
        index = json.loads(paths["alignment_index"].read_text())
        if index.get("format") != INDEX_FORMAT or index.get("guards", {}).get("reward_read") is not False:
            raise RuntimeError("invalid v440 train-only alignment index")
        if index.get("train_episodes") != manifest.get("train_episodes") or len(index.get("right_train_episodes", [])) != 15:
            raise RuntimeError("v440 train-only episode binding mismatch")
        self.beta = {phase: np.asarray(index["beta"][phase], dtype=np.float64) for phase in PHASES}
        for phase, value in self.beta.items():
            if value.shape != (8, 3) or not np.isfinite(value).all() or np.abs(value).max() > 1.0:
                raise RuntimeError(f"invalid v440 beta: {phase}")
        v436_root = root / manifest["v436_release"]
        v436 = json.loads(paths["v436_manifest"].read_text())
        if v436.get("format") != V436_FORMAT or v436.get("formal_candidate_authorized") is not False:
            raise RuntimeError("v440 requires the frozen v432-step25 teacher release")
        for key in ("parent_right", "candidate_right"):
            model = v436_root / v436[key] / "model.pt"
            if not model.is_file() or _sha256(model) != v436["model_sha256"][key]:
                raise RuntimeError(f"v440 teacher hash mismatch: {key}")
        self.v169 = Track2V169ArmRoutedRuntime(root / manifest["v169_release"], root / manifest["v169_library"], device)
        self.parent = Track2AutoregressiveUNet(v436_root / v436["parent_right"], device)
        self.candidate = Track2AutoregressiveUNet(v436_root / v436["candidate_right"], device)
        self.last_decisions: list[dict] = []

    @staticmethod
    def gate_decision(history_actions, future_actions, instruction):
        return gate_decision(history_actions, future_actions, instruction)

    def predict_batch_with_baseline(self, context_frames, history_actions, future_actions, seeds, instructions):
        baseline = self.v169.predict_batch(context_frames, history_actions, future_actions, seeds, instructions)
        decisions = [gate_decision(h, f, t) for h, f, t in zip(history_actions, future_actions, instructions)]
        enabled = np.asarray([row["gate"] for row in decisions], dtype=bool)
        output = baseline.copy()
        if enabled.any():
            indices = np.flatnonzero(enabled)
            texts = [instructions[index] for index in indices]
            selected_seeds = np.asarray(seeds)[enabled]
            parent = self.parent.predict_batch(np.asarray(context_frames)[enabled], np.asarray(history_actions)[enabled], np.asarray(future_actions)[enabled], selected_seeds, texts)
            candidate = self.candidate.predict_batch(np.asarray(context_frames)[enabled], np.asarray(history_actions)[enabled], np.asarray(future_actions)[enabled], selected_seeds, texts)
            for local, index in enumerate(indices):
                output[index] = apply_aligned_projection(baseline[index], candidate[local], parent[local], self.beta[decisions[index]["phase"]])
        self.last_decisions = decisions
        return baseline, output, decisions

    def predict_with_baseline(self, context_frames, history_actions, future_actions, seed, instruction):
        baseline, output, decisions = self.predict_batch_with_baseline(
            np.asarray(context_frames)[None], np.asarray(history_actions)[None], np.asarray(future_actions)[None], np.asarray([seed]), [instruction]
        )
        return baseline[0], output[0], decisions[0]

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        return self.predict_with_baseline(context_frames, history_actions, future_actions, seed, instruction)[1]

    def predict_batch(self, context_frames, history_actions, future_actions, seeds, instructions):
        return self.predict_batch_with_baseline(context_frames, history_actions, future_actions, seeds, instructions)[1]
