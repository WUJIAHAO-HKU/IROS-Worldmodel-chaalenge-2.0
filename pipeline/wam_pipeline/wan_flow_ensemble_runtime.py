"""Portable Wan/direct-flow ensemble runtime for the Track 2 RGB contract."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .profile import ACTION_DIM, CONTEXT_ACTIONS, CONTEXT_FRAMES, PREDICTION_FRAMES


FORMAT = "track2-wan-direct-flow-ensemble-v1"
_WAN_FILES = ("track2_wan_lora.pt", "training_manifest.json", "action_normalization.npz")
_FLOW_FILES = ("model.pt", "training_manifest.json", "action_normalization.npz", "track2_direct_flow_unet_config.npz")
_BASE_MANIFEST = "track2_wan_weight_manifest.json"


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


def wan_base_identity(base_model: str | Path) -> dict[str, str]:
    """Read the verified upstream base identity without rehashing 27GB at startup."""
    base = Path(base_model).resolve()
    manifest = base / _BASE_MANIFEST
    if not manifest.is_file():
        raise RuntimeError(f"Wan base is missing its verified weight manifest: {manifest}")
    try:
        contents = json.loads(manifest.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError("Wan base weight manifest is not valid JSON") from exc
    repository = contents.get("repository")
    revision = contents.get("revision")
    if not isinstance(repository, str) or not repository or not isinstance(revision, str) or len(revision) != 40:
        raise RuntimeError("Wan base weight manifest has no immutable repository revision")
    return {
        "repository": repository,
        "revision": revision,
        "weight_manifest_sha256": _sha256(manifest),
    }


def load_ensemble_config(checkpoint_dir: str | Path) -> dict[str, Any]:
    """Load and verify the small, self-contained ensemble manifest."""
    root = Path(checkpoint_dir).resolve()
    config_path = root / "ensemble_config.json"
    if not config_path.is_file():
        raise RuntimeError("Wan/direct-flow ensemble requires ensemble_config.json")
    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError("Wan/direct-flow ensemble config is not valid JSON") from exc
    if not isinstance(config, dict) or config.get("format") != FORMAT:
        raise RuntimeError("unsupported Wan/direct-flow ensemble format")

    weights = (config.get("wan_weight"), config.get("direct_flow_weight"))
    if any(not isinstance(weight, (int, float)) or not math.isfinite(float(weight)) for weight in weights):
        raise RuntimeError("ensemble weights must be finite numbers")
    if any(float(weight) < 0.0 or float(weight) > 1.0 for weight in weights) or not math.isclose(
        float(weights[0]) + float(weights[1]), 1.0, rel_tol=0.0, abs_tol=1e-8
    ):
        raise RuntimeError("ensemble weights must be in [0, 1] and sum to one")

    wan = config.get("wan")
    flow = config.get("direct_flow")
    if not isinstance(wan, dict) or not isinstance(flow, dict):
        raise RuntimeError("ensemble config is missing a Wan or direct-flow component")
    if not isinstance(wan.get("inference_steps"), int) or int(wan["inference_steps"]) < 1:
        raise RuntimeError("ensemble Wan inference_steps must be a positive integer")
    if wan.get("inference_solver") not in {"euler", "heun"}:
        raise RuntimeError("ensemble Wan inference_solver must be euler or heun")
    base = wan.get("base_model")
    if not isinstance(base, dict):
        raise RuntimeError("ensemble Wan component is missing its immutable base identity")
    if (
        not isinstance(base.get("repository"), str)
        or not base["repository"]
        or not isinstance(base.get("revision"), str)
        or len(base["revision"]) != 40
        or not isinstance(base.get("weight_manifest_sha256"), str)
        or len(base["weight_manifest_sha256"]) != 64
    ):
        raise RuntimeError("ensemble Wan base identity is malformed")

    for component, required_files, name in ((wan, _WAN_FILES, "wan"), (flow, _FLOW_FILES, "direct_flow")):
        directory = _component_path(root, component, name)
        hashes = component.get("sha256")
        if not isinstance(hashes, dict):
            raise RuntimeError(f"ensemble component {name!r} is missing artifact hashes")
        if set(hashes) != set(required_files):
            raise RuntimeError(f"ensemble component {name!r} does not identify all required artifacts")
        for filename in required_files:
            expected = hashes[filename]
            path = directory / filename
            if not isinstance(expected, str) or len(expected) != 64 or not path.is_file():
                raise RuntimeError(f"ensemble component artifact is missing or malformed: {path}")
            if _sha256(path) != expected:
                raise RuntimeError(f"ensemble component artifact hash mismatch: {path}")
    return config


def ensemble_artifact_sha256(checkpoint_dir: str | Path) -> str:
    """Hash the config and every deployable file in deterministic order."""
    root = Path(checkpoint_dir).resolve()
    config = load_ensemble_config(root)
    digest = hashlib.sha256()
    files: list[Path] = [root / "ensemble_config.json"]
    for component_name, required_files in (("wan", _WAN_FILES), ("direct_flow", _FLOW_FILES)):
        directory = _component_path(root, config[component_name], component_name)
        files.extend(directory / filename for filename in required_files)
    for path in files:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def blend_predictions(
    wan_prediction: np.ndarray,
    direct_flow_prediction: np.ndarray,
    wan_weight: float,
    direct_flow_weight: float,
) -> np.ndarray:
    """Blend two audited RGB outputs without changing the external contract."""
    expected = (PREDICTION_FRAMES, 256, 256, 3)
    if wan_prediction.shape != expected or direct_flow_prediction.shape != expected:
        raise RuntimeError("ensemble members returned an invalid Track 2 prediction shape")
    if wan_prediction.dtype != np.uint8 or direct_flow_prediction.dtype != np.uint8:
        raise RuntimeError("ensemble members must return uint8 RGB predictions")
    blended = np.rint(
        wan_prediction.astype(np.float32) * float(wan_weight)
        + direct_flow_prediction.astype(np.float32) * float(direct_flow_weight)
    )
    return blended.clip(0, 255).astype(np.uint8)


class Track2WanFlowEnsemble:
    """One native Wan rollout plus one direct-flow rollout, fused at RGB output."""

    def __init__(self, checkpoint_dir: str | Path, base_model: str | Path | None, device: str = "cuda") -> None:
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.config = load_ensemble_config(self.checkpoint_dir)
        if not base_model:
            raise ValueError("Wan/direct-flow ensemble requires the local verified Wan base model")
        self.base_model = Path(base_model).resolve()
        expected_base = self.config.get("wan", {}).get("base_model")
        actual_base = wan_base_identity(self.base_model)
        if not isinstance(expected_base, dict) or expected_base != actual_base:
            raise RuntimeError("Wan base identity does not match the packaged ensemble")
        self.device = device
        self._wan = None
        self._direct_flow = None

    @property
    def metadata(self) -> dict[str, object]:
        wan = self.config["wan"]
        return {
            "format": FORMAT,
            "wan_weight": float(self.config["wan_weight"]),
            "direct_flow_weight": float(self.config["direct_flow_weight"]),
            "wan_inference": {"steps": int(wan["inference_steps"]), "solver": wan["inference_solver"]},
            "wan_base_model": wan_base_identity(self.base_model),
            "artifact_sha256": ensemble_artifact_sha256(self.checkpoint_dir),
        }

    def _load_members(self) -> None:
        if self._wan is not None:
            return
        from .direct_flow_unet_runtime import Track2DirectFlowUNet
        from .track2_wan_runtime import Track2WanRuntime

        wan = self.config["wan"]
        self._wan = Track2WanRuntime(
            _component_path(self.checkpoint_dir, wan, "wan"),
            base_model=self.base_model,
            device=self.device,
            inference_steps=int(wan["inference_steps"]),
            inference_solver=str(wan["inference_solver"]),
        )
        self._direct_flow = Track2DirectFlowUNet(
            _component_path(self.checkpoint_dir, self.config["direct_flow"], "direct_flow"), self.device
        )

    def predict(self, context_frames, history_actions, future_actions, seed: int, instruction: str | None):
        del instruction
        if context_frames.shape != (CONTEXT_FRAMES, 256, 256, 3) or context_frames.dtype != np.uint8:
            raise ValueError("context_frames must be [5,256,256,3] uint8")
        if history_actions.shape != (CONTEXT_ACTIONS, ACTION_DIM) or future_actions.shape != (PREDICTION_FRAMES, ACTION_DIM):
            raise ValueError("actions must be [4,14] history and [8,14] future")
        self._load_members()
        wan_prediction = self._wan.predict(context_frames, history_actions, future_actions, seed, None)
        flow_prediction = self._direct_flow.predict(context_frames, history_actions, future_actions, seed, None)
        return blend_predictions(
            wan_prediction,
            flow_prediction,
            float(self.config["wan_weight"]),
            float(self.config["direct_flow_weight"]),
        )
