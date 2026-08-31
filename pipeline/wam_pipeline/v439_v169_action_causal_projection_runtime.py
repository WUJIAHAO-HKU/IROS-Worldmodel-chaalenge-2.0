"""V439: v169 RGB baseline plus a strictly gated right-action causal delta."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .autoregressive_unet_runtime import Track2AutoregressiveUNet
from .v169_arm_routed_runtime import Track2V169ArmRoutedRuntime


FORMAT = "track2-v439-v169-action-causal-projection-release-v1"
INDEX_FORMAT = "strict-track2-v439-public-right-action-projection-index-v1"
V436_FORMAT = "track2-v436-v432-step25-diagnostic-release-v1"
MU = np.asarray([-0.330, -1.425, -1.563, 1.617, 0.494, 0.779], dtype=np.float64)
ALPHA = np.asarray([0.0, 0.0, 0.03661165, 0.125, 0.125, 0.03661165, 0.0, 0.0], dtype=np.float64)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _explicit_right(instruction: str | None) -> bool:
    text = str(instruction or "").lower()
    return "right arm" in text and "left arm" not in text


def _joint_path(actions: np.ndarray, begin: int) -> float:
    return float(np.linalg.norm(np.diff(actions[:, begin : begin + 6], axis=0), axis=1).sum())


def gate_decision(history_actions: np.ndarray, future_actions: np.ndarray, instruction: str | None) -> dict:
    history = np.asarray(history_actions, dtype=np.float32)
    future = np.asarray(future_actions, dtype=np.float32)
    if history.ndim != 2 or future.ndim != 2 or history.shape[1] != 14 or future.shape != (8, 14):
        raise ValueError(f"v439 expects history[T,14] and future[8,14], got {history.shape}, {future.shape}")
    actions = np.concatenate((history, future), axis=0)
    left_path = _joint_path(actions, 0)
    right_path = _joint_path(actions, 7)
    explicit_right = _explicit_right(instruction)
    right_dominant = right_path > left_path
    right_gripper = np.concatenate((history[-1:, 13], future[:, 13]))
    closed = right_gripper < 0.5
    close_indices = np.flatnonzero(closed)
    right_closed = bool(close_indices.size and np.all(closed[close_indices[0] :]))
    right_delta = future[-1, 7:13] - history[-1, 7:13]
    delta_dot_mu = float(np.dot(right_delta, MU))
    gate = bool(explicit_right and right_dominant and right_closed and delta_dot_mu > 0.0)
    if not explicit_right:
        reason = "prompt_not_explicit_right"
    elif not right_dominant:
        reason = "right_path_not_greater_than_left"
    elif not right_closed:
        reason = "right_gripper_never_closes_or_does_not_remain_held"
    elif delta_dot_mu <= 0.0:
        reason = "postclose_delta_projection_not_positive"
    else:
        reason = "right_postclose_projection"
    return {
        "gate": gate,
        "explicit_right": explicit_right,
        "right_path": right_path,
        "left_path": left_path,
        "right_closed": right_closed,
        "delta_dot_mu": delta_dot_mu,
        "reason": reason,
    }


def clipped_rgb_delta(candidate: np.ndarray, parent: np.ndarray) -> np.ndarray:
    candidate_array = np.asarray(candidate)
    parent_array = np.asarray(parent)
    if candidate_array.shape != parent_array.shape or candidate_array.shape[-1] != 3:
        raise ValueError("v439 teacher RGB shapes do not match")
    return np.clip(
        candidate_array.astype(np.int16) - parent_array.astype(np.int16), -8, 8
    ).astype(np.int16)


def apply_projection(baseline: np.ndarray, candidate: np.ndarray, parent: np.ndarray, enabled: bool) -> np.ndarray:
    baseline_array = np.asarray(baseline)
    if baseline_array.dtype != np.uint8 or baseline_array.ndim != 4 or baseline_array.shape[0] != 8 or baseline_array.shape[-1] != 3:
        raise ValueError(f"v439 expects baseline uint8 RGB[8,H,W,3], got {baseline_array.shape}/{baseline_array.dtype}")
    if not enabled:
        return baseline_array.copy()
    delta = clipped_rgb_delta(candidate, parent).astype(np.float32)
    output = np.rint(
        baseline_array.astype(np.float32) + ALPHA[:, None, None, None] * delta
    )
    output = np.clip(output, 0, 255).astype(np.uint8)
    # Explicit assignment makes the bit-exact frame contract independent of
    # floating-point signed-zero or rounding behavior.
    output[ALPHA == 0.0] = baseline_array[ALPHA == 0.0]
    return output


class Track2V439V169ActionCausalProjection:
    """Never replace v169; add the frozen delta only for gated right requests."""

    def __init__(self, release_dir: str | Path, device: str = "cuda") -> None:
        root = Path(release_dir).resolve()
        manifest_path = root / "v439_action_causal_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format") != FORMAT:
            raise RuntimeError("unsupported v439 release")
        if manifest.get("official_reward_runtime_used") is not False:
            raise RuntimeError("v439 runtime must not use official reward")
        paths = {
            "projection_index": root / manifest["projection_index"],
            "v169_manifest": root / manifest["v169_release"] / "v169_arm_routed_manifest.json",
            "v436_manifest": root / manifest["v436_release"] / "v436_diagnostic_manifest.json",
        }
        for key, path in paths.items():
            if not path.is_file() or _sha256(path) != manifest["sha256"][key]:
                raise RuntimeError(f"v439 release hash mismatch: {key}")
        index = json.loads(paths["projection_index"].read_text())
        if index.get("format") != INDEX_FORMAT:
            raise RuntimeError("wrong v439 projection index")
        projection = index.get("projection", {})
        if projection.get("mu_right6d") != MU.astype(float).tolist() or projection.get("alpha_8") != ALPHA.astype(float).tolist():
            raise RuntimeError("v439 projection constants drifted")
        if index.get("guards", {}).get("reward_read") is not False or index.get("guards", {}).get("outcome_read") is not False:
            raise RuntimeError("v439 projection index violates the data boundary")
        v436_root = root / manifest["v436_release"]
        v436 = json.loads(paths["v436_manifest"].read_text())
        if v436.get("format") != V436_FORMAT or v436.get("formal_candidate_authorized") is not False:
            raise RuntimeError("v439 requires the frozen v432-step25 diagnostic release")
        for key, relative in (("parent_right", v436["parent_right"]), ("candidate_right", v436["candidate_right"])):
            model = v436_root / relative / "model.pt"
            if not model.is_file() or _sha256(model) != v436["model_sha256"][key]:
                raise RuntimeError(f"v439 teacher hash mismatch: {key}")
        self.v169 = Track2V169ArmRoutedRuntime(
            root / manifest["v169_release"], root / manifest["v169_library"], device
        )
        self.parent = Track2AutoregressiveUNet(v436_root / v436["parent_right"], device)
        self.candidate = Track2AutoregressiveUNet(v436_root / v436["candidate_right"], device)
        self.last_decisions: list[dict] = []

    @staticmethod
    def gate_decision(history_actions, future_actions, instruction):
        return gate_decision(history_actions, future_actions, instruction)

    def predict_batch_with_baseline(self, context_frames, history_actions, future_actions, seeds, instructions):
        baseline = self.v169.predict_batch(context_frames, history_actions, future_actions, seeds, instructions)
        decisions = [
            gate_decision(history, future, instruction)
            for history, future, instruction in zip(history_actions, future_actions, instructions)
        ]
        enabled = np.asarray([row["gate"] for row in decisions], dtype=bool)
        output = baseline.copy()
        if enabled.any():
            indices = np.flatnonzero(enabled)
            selected_instructions = [instructions[index] for index in indices]
            selected_seeds = np.asarray(seeds)[enabled]
            parent = self.parent.predict_batch(
                np.asarray(context_frames)[enabled], np.asarray(history_actions)[enabled],
                np.asarray(future_actions)[enabled], selected_seeds, selected_instructions,
            )
            candidate = self.candidate.predict_batch(
                np.asarray(context_frames)[enabled], np.asarray(history_actions)[enabled],
                np.asarray(future_actions)[enabled], selected_seeds, selected_instructions,
            )
            for local, index in enumerate(indices):
                output[index] = apply_projection(baseline[index], candidate[local], parent[local], True)
        self.last_decisions = decisions
        return baseline, output, decisions

    def predict_with_baseline(self, context_frames, history_actions, future_actions, seed, instruction):
        baseline, output, decisions = self.predict_batch_with_baseline(
            np.asarray(context_frames)[None], np.asarray(history_actions)[None],
            np.asarray(future_actions)[None], np.asarray([seed]), [instruction],
        )
        return baseline[0], output[0], decisions[0]

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        return self.predict_with_baseline(context_frames, history_actions, future_actions, seed, instruction)[1]

    def predict_batch(self, context_frames, history_actions, future_actions, seeds, instructions):
        return self.predict_batch_with_baseline(context_frames, history_actions, future_actions, seeds, instructions)[1]
