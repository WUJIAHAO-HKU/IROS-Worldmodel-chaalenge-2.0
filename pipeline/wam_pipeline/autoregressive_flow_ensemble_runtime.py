"""Observable-motion-gated autoregressive/direct-flow Track 2 runtime."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .profile import ACTION_DIM, CONTEXT_ACTIONS, CONTEXT_FRAMES, PREDICTION_FRAMES


FORMAT = "track2-autoregressive-direct-flow-motion-gated-v1"
_AUTOREGRESSIVE_FILES = (
    "model.pt",
    "training_manifest.json",
    "action_normalization.npz",
    "track2_autoregressive_unet_config.npz",
)
_FLOW_FILES = (
    "model.pt",
    "training_manifest.json",
    "action_normalization.npz",
    "track2_direct_flow_unet_config.npz",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _component_path(root: Path, component: dict[str, Any], name: str) -> Path:
    relative = component.get("directory")
    if not isinstance(relative, str) or not relative:
        raise RuntimeError(f"ensemble component {name!r} has no relative directory")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"ensemble component {name!r} escapes its checkpoint directory") from exc
    if not path.is_dir():
        raise RuntimeError(f"ensemble component directory is missing: {path}")
    return path


def load_ensemble_config(checkpoint_dir: str | Path) -> dict[str, Any]:
    root = Path(checkpoint_dir).resolve()
    config_path = root / "ensemble_config.json"
    if not config_path.is_file():
        raise RuntimeError("motion-gated ensemble requires ensemble_config.json")
    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError("motion-gated ensemble config is not valid JSON") from exc
    if not isinstance(config, dict) or config.get("format") != FORMAT:
        raise RuntimeError("unsupported motion-gated ensemble format")
    if config.get("context_motion_statistic") != "last":
        raise RuntimeError("motion-gated ensemble only supports the last observed transition")
    threshold = config.get("context_motion_threshold")
    if not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)) or float(threshold) < 0:
        raise RuntimeError("context motion threshold must be a finite non-negative number")
    weights = (
        config.get("low_motion_autoregressive_weight"),
        config.get("low_motion_direct_flow_weight"),
    )
    if any(not isinstance(weight, (int, float)) or not math.isfinite(float(weight)) for weight in weights):
        raise RuntimeError("low-motion ensemble weights must be finite numbers")
    if any(not 0 <= float(weight) <= 1 for weight in weights) or not math.isclose(
        float(weights[0]) + float(weights[1]), 1.0, rel_tol=0.0, abs_tol=1e-8
    ):
        raise RuntimeError("low-motion ensemble weights must be in [0,1] and sum to one")

    autoregressive = config.get("autoregressive")
    direct_flow = config.get("direct_flow")
    if not isinstance(autoregressive, dict) or not isinstance(direct_flow, dict):
        raise RuntimeError("ensemble config is missing a model component")
    for component, files, name in (
        (autoregressive, _AUTOREGRESSIVE_FILES, "autoregressive"),
        (direct_flow, _FLOW_FILES, "direct_flow"),
    ):
        directory = _component_path(root, component, name)
        hashes = component.get("sha256")
        if not isinstance(hashes, dict) or set(hashes) != set(files):
            raise RuntimeError(f"ensemble component {name!r} does not identify all required artifacts")
        for filename in files:
            path = directory / filename
            expected = hashes[filename]
            if not path.is_file() or not isinstance(expected, str) or len(expected) != 64:
                raise RuntimeError(f"ensemble component artifact is missing or malformed: {path}")
            if _sha256(path) != expected:
                raise RuntimeError(f"ensemble component artifact hash mismatch: {path}")
    return config


def ensemble_artifact_sha256(checkpoint_dir: str | Path) -> str:
    root = Path(checkpoint_dir).resolve()
    config = load_ensemble_config(root)
    files = [root / "ensemble_config.json"]
    for name, required in (
        ("autoregressive", _AUTOREGRESSIVE_FILES),
        ("direct_flow", _FLOW_FILES),
    ):
        directory = _component_path(root, config[name], name)
        files.extend(directory / filename for filename in required)
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def last_context_motion(context_frames: np.ndarray) -> float:
    if context_frames.shape != (CONTEXT_FRAMES, 256, 256, 3) or context_frames.dtype != np.uint8:
        raise ValueError("context_frames must be [5,256,256,3] uint8")
    difference = context_frames[-1].astype(np.int16) - context_frames[-2].astype(np.int16)
    return float(np.abs(difference).mean(dtype=np.float64) / 255.0)


def blend_predictions(
    autoregressive_prediction: np.ndarray,
    direct_flow_prediction: np.ndarray,
    autoregressive_weight: float,
    direct_flow_weight: float,
) -> np.ndarray:
    expected = (PREDICTION_FRAMES, 256, 256, 3)
    if autoregressive_prediction.shape != expected or direct_flow_prediction.shape != expected:
        raise RuntimeError("ensemble members returned an invalid Track 2 prediction shape")
    if autoregressive_prediction.dtype != np.uint8 or direct_flow_prediction.dtype != np.uint8:
        raise RuntimeError("ensemble members must return uint8 RGB predictions")
    blended = np.rint(
        autoregressive_prediction.astype(np.float32) * float(autoregressive_weight)
        + direct_flow_prediction.astype(np.float32) * float(direct_flow_weight)
    )
    return blended.clip(0, 255).astype(np.uint8)


class Track2AutoregressiveFlowMotionGate:
    """Use autoregression for observed motion and a two-small-model blend otherwise."""

    def __init__(self, checkpoint_dir: str | Path, device: str = "cuda") -> None:
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.config = load_ensemble_config(self.checkpoint_dir)
        self.device = device
        self._autoregressive = None
        self._direct_flow = None

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "format": FORMAT,
            "context_motion_statistic": "last",
            "context_motion_threshold": float(self.config["context_motion_threshold"]),
            "low_motion_autoregressive_weight": float(
                self.config["low_motion_autoregressive_weight"]
            ),
            "low_motion_direct_flow_weight": float(self.config["low_motion_direct_flow_weight"]),
            "artifact_sha256": ensemble_artifact_sha256(self.checkpoint_dir),
        }

    def _load_autoregressive(self) -> None:
        if self._autoregressive is None:
            from .autoregressive_unet_runtime import Track2AutoregressiveUNet

            self._autoregressive = Track2AutoregressiveUNet(
                _component_path(
                    self.checkpoint_dir,
                    self.config["autoregressive"],
                    "autoregressive",
                ),
                self.device,
            )

    def _load_direct_flow(self) -> None:
        if self._direct_flow is None:
            from .direct_flow_unet_runtime import Track2DirectFlowUNet

            self._direct_flow = Track2DirectFlowUNet(
                _component_path(self.checkpoint_dir, self.config["direct_flow"], "direct_flow"),
                self.device,
            )

    def predict(self, context_frames, history_actions, future_actions, seed: int, instruction: str | None):
        del instruction
        motion = last_context_motion(context_frames)
        if history_actions.shape != (CONTEXT_ACTIONS, ACTION_DIM):
            raise ValueError("history_actions must be [4,14]")
        if future_actions.shape != (PREDICTION_FRAMES, ACTION_DIM):
            raise ValueError("future_actions must be [8,14]")
        self._load_autoregressive()
        autoregressive_prediction = self._autoregressive.predict(
            context_frames, history_actions, future_actions, seed, None
        )
        if motion >= float(self.config["context_motion_threshold"]):
            return autoregressive_prediction
        self._load_direct_flow()
        direct_flow_prediction = self._direct_flow.predict(
            context_frames, history_actions, future_actions, seed, None
        )
        return blend_predictions(
            autoregressive_prediction,
            direct_flow_prediction,
            float(self.config["low_motion_autoregressive_weight"]),
            float(self.config["low_motion_direct_flow_weight"]),
        )
