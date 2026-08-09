"""Packaged runtime for the learned AR/direct-flow local fusion model."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .profile import ACTION_DIM, CONTEXT_ACTIONS, CONTEXT_FRAMES, PREDICTION_FRAMES


FORMAT = "track2-local-motion-texture-fusion-ensemble-v1"
COMPONENT_FILES = {
    "autoregressive": ("model.pt", "training_manifest.json", "action_normalization.npz", "track2_autoregressive_unet_config.npz"),
    "direct_flow": ("model.pt", "training_manifest.json", "action_normalization.npz", "track2_direct_flow_unet_config.npz"),
    "fusion": ("model.pt", "training_manifest.json", "action_normalization.npz", "local_fusion_config.npz"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _component(root: Path, config: dict, name: str) -> Path:
    component = config.get(name)
    if not isinstance(component, dict) or component.get("directory") != name:
        raise RuntimeError(f"invalid local-fusion component: {name}")
    directory = (root / name).resolve()
    directory.relative_to(root.resolve())
    hashes = component.get("sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(COMPONENT_FILES[name]):
        raise RuntimeError(f"incomplete local-fusion hashes: {name}")
    for filename in COMPONENT_FILES[name]:
        path = directory / filename
        if not path.is_file() or hashes[filename] != _sha256(path):
            raise RuntimeError(f"local-fusion artifact hash mismatch: {path}")
    return directory


def load_ensemble_config(checkpoint_dir: str | Path) -> dict:
    root = Path(checkpoint_dir).resolve()
    config = json.loads((root / "ensemble_config.json").read_text())
    if config.get("format") != FORMAT:
        raise RuntimeError("unsupported local motion/texture fusion format")
    for name in COMPONENT_FILES:
        _component(root, config, name)
    return config


def ensemble_artifact_sha256(checkpoint_dir: str | Path) -> str:
    root = Path(checkpoint_dir).resolve()
    config = load_ensemble_config(root)
    paths = [root / "ensemble_config.json"]
    for name, filenames in COMPONENT_FILES.items():
        directory = _component(root, config, name)
        paths.extend(directory / filename for filename in filenames)
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(_sha256(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


class Track2LocalMotionTextureFusion:
    def __init__(self, checkpoint_dir: str | Path, device: str = "cuda") -> None:
        import torch

        from .autoregressive_unet_runtime import Track2AutoregressiveUNet
        from .direct_flow_unet_runtime import Track2DirectFlowUNet
        from .local_fusion import LocalMotionTextureFusion

        self.root = Path(checkpoint_dir).resolve()
        self.config = load_ensemble_config(self.root)
        self.device = torch.device(device)
        self.torch = torch
        self.autoregressive = Track2AutoregressiveUNet(_component(self.root, self.config, "autoregressive"), device)
        self.direct_flow = Track2DirectFlowUNet(_component(self.root, self.config, "direct_flow"), device)
        fusion_dir = _component(self.root, self.config, "fusion")
        model_config = np.load(fusion_dir / "local_fusion_config.npz", allow_pickle=False)
        self.model = LocalMotionTextureFusion(int(model_config["base_channels"]), float(model_config["residual_scale"])).to(self.device).eval()
        state = torch.load(fusion_dir / "model.pt", map_location="cpu", weights_only=True)
        if state.get("format") != "track2-local-motion-texture-fusion-v1":
            raise RuntimeError("unsupported local fusion model")
        self.model.load_state_dict(state["state_dict"], strict=True)
        normalization = np.load(fusion_dir / "action_normalization.npz", allow_pickle=False)
        self.mean = np.asarray(normalization["mean"], dtype=np.float32)
        self.std = np.asarray(normalization["std"], dtype=np.float32)

    @property
    def metadata(self) -> dict[str, object]:
        return {"format": FORMAT, "artifact_sha256": ensemble_artifact_sha256(self.root)}

    def predict(self, context_frames, history_actions, future_actions, seed: int, instruction: str | None):
        del instruction
        if context_frames.shape != (CONTEXT_FRAMES, 256, 256, 3) or context_frames.dtype != np.uint8:
            raise ValueError("context_frames must be [5,256,256,3] uint8")
        if history_actions.shape != (CONTEXT_ACTIONS, ACTION_DIM) or future_actions.shape != (PREDICTION_FRAMES, ACTION_DIM):
            raise ValueError("actions must be [4,14] history and [8,14] future")
        first = self.autoregressive.predict(context_frames, history_actions, future_actions, seed, None)
        second = self.direct_flow.predict(context_frames, history_actions, future_actions, seed, None)
        torch = self.torch
        context = torch.from_numpy(context_frames[-1:].copy()).permute(0, 3, 1, 2).unsqueeze(0).to(self.device).float().div(255.0)
        first_tensor = torch.from_numpy(first.copy()).permute(0, 3, 1, 2).unsqueeze(0).to(self.device).float().div(255.0)
        second_tensor = torch.from_numpy(second.copy()).permute(0, 3, 1, 2).unsqueeze(0).to(self.device).float().div(255.0)
        actions = torch.from_numpy(np.ascontiguousarray((future_actions - self.mean) / self.std)).unsqueeze(0).to(self.device)
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            prediction, _, _ = self.model(context, first_tensor, second_tensor, actions.float())
        return prediction.clamp(0, 1).mul(255).round().to(torch.uint8).squeeze(0).permute(0, 2, 3, 1).cpu().numpy().copy()
