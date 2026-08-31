"""V443: train-only spatial LOEO teacher correction on original v169 RGB."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .autoregressive_unet_runtime import Track2AutoregressiveUNet
from .v169_arm_routed_runtime import Track2V169ArmRoutedRuntime
from .v442_v169_close_aligned_projection_runtime import clipped_rgb_delta, gate_decision


FORMAT = "track2-v443-v169-close-spatial-projection-release-v1"
INDEX_FORMAT = "strict-track2-v443-close-trainonly-spatial-alignment-index-v1"
V436_FORMAT = "track2-v436-v432-step25-diagnostic-release-v1"
PROTECTED_FRAMES = (0, 1, 6, 7)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_spatial_projection(
    baseline: np.ndarray,
    candidate: np.ndarray,
    parent: np.ndarray,
    beta_map: np.ndarray | None,
) -> np.ndarray:
    baseline = np.asarray(baseline)
    if baseline.dtype != np.uint8 or baseline.ndim != 4 or baseline.shape[0] != 8 or baseline.shape[-1] != 3:
        raise ValueError(f"v443 expects baseline uint8 RGB[8,H,W,3], got {baseline.shape}/{baseline.dtype}")
    if beta_map is None:
        return baseline.copy()
    beta = np.asarray(beta_map, dtype=np.float32)
    if beta.shape != baseline.shape or not np.isfinite(beta).all() or np.abs(beta).max(initial=0.0) > 1.0:
        raise ValueError("v443 beta map must match RGB[8,H,W,3], be finite, and have abs<=1")
    if np.any(beta[list(PROTECTED_FRAMES)] != 0.0):
        raise ValueError("v443 protected frame coefficients must be exactly zero")
    delta = clipped_rgb_delta(candidate, parent).astype(np.float32)
    output = np.rint(baseline.astype(np.float32) + beta * delta)
    return np.clip(output, 0, 255).astype(np.uint8)


class Track2V443V169CloseSpatialProjection:
    def __init__(self, release_dir: str | Path, device: str = "cuda") -> None:
        root = Path(release_dir).resolve()
        manifest_path = root / "v443_close_spatial_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format") != FORMAT or manifest.get("official_reward_runtime_used") is not False:
            raise RuntimeError("unsupported or reward-coupled v443 release")
        paths = {
            "alignment_index": root / manifest["alignment_index"],
            "v169_manifest": root / manifest["v169_release"] / "v169_arm_routed_manifest.json",
            "v436_manifest": root / manifest["v436_release"] / "v436_diagnostic_manifest.json",
        }
        for key, path in paths.items():
            if not path.is_file() or _sha256(path) != manifest["sha256"][key]:
                raise RuntimeError(f"v443 release hash mismatch: {key}")
        with np.load(paths["alignment_index"], allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            beta = np.asarray(archive["beta_map"], dtype=np.float32)
        if metadata.get("format") != INDEX_FORMAT or metadata.get("guards", {}).get("reward_read") is not False:
            raise RuntimeError("invalid v443 train-only alignment index")
        if metadata.get("train_episodes") != manifest.get("train_episodes") or len(metadata.get("right_train_episodes", [])) != 15:
            raise RuntimeError("v443 train-only episode binding mismatch")
        if beta.ndim != 4 or beta.shape[0] != 8 or beta.shape[-1] != 3:
            raise RuntimeError(f"invalid v443 beta shape: {beta.shape}")
        if not np.isfinite(beta).all() or np.abs(beta).max(initial=0.0) > 1.0 or np.any(beta[list(PROTECTED_FRAMES)] != 0.0):
            raise RuntimeError("invalid or unprotected v443 beta map")
        if int(metadata.get("loeo", {}).get("active_pixels", 0)) < int(metadata.get("loeo", {}).get("active_pixels_min", 0)):
            raise RuntimeError("v443 active-pixel guard failed")
        if not float(metadata.get("loeo", {}).get("aggregate_sse_ratio", float("inf"))) < 1.0:
            raise RuntimeError("v443 train LOEO aggregate improvement guard failed")
        self.beta_map = beta
        v436_root = root / manifest["v436_release"]
        v436 = json.loads(paths["v436_manifest"].read_text())
        if v436.get("format") != V436_FORMAT or v436.get("formal_candidate_authorized") is not False:
            raise RuntimeError("v443 requires frozen v432-step25 teacher release")
        for key in ("parent_right", "candidate_right"):
            model = v436_root / v436[key] / "model.pt"
            if not model.is_file() or _sha256(model) != v436["model_sha256"][key]:
                raise RuntimeError(f"v443 teacher hash mismatch: {key}")
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
                output[index] = apply_spatial_projection(baseline[index], candidate[local], parent[local], self.beta_map)
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
