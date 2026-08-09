"""Self-contained runtime for structure-gated local high-frequency refinement."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .profile import ACTION_DIM, CONTEXT_ACTIONS, CONTEXT_FRAMES, PREDICTION_FRAMES


FORMAT = "track2-structure-gated-local-fusion-ensemble-v1"
FILES = {
    "structure_parent": ("model.pt", "training_manifest.json", "action_normalization.npz", "track2_autoregressive_structure_unet_config.npz"),
    "refiner": ("model.pt", "training_manifest.json", "action_normalization.npz", "structure_refiner_config.npz"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _component(root: Path, config: dict, name: str) -> Path:
    entry = config.get(name)
    if not isinstance(entry, dict) or entry.get("directory") != name:
        raise RuntimeError(f"invalid structure-refiner component: {name}")
    directory = (root / name).resolve()
    directory.relative_to(root.resolve())
    hashes = entry.get("sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(FILES[name]):
        raise RuntimeError(f"incomplete structure-refiner hashes: {name}")
    for filename in FILES[name]:
        path = directory / filename
        if not path.is_file() or hashes[filename] != _sha256(path):
            raise RuntimeError(f"structure-refiner artifact hash mismatch: {path}")
    return directory


def load_ensemble_config(checkpoint_dir: str | Path) -> dict:
    from .local_fusion_runtime import ensemble_artifact_sha256

    root = Path(checkpoint_dir).resolve()
    config = json.loads((root / "ensemble_config.json").read_text())
    if config.get("format") != FORMAT:
        raise RuntimeError("unsupported structure-gated fusion format")
    for name in FILES:
        _component(root, config, name)
    baseline = (root / "baseline").resolve()
    baseline.relative_to(root)
    if ensemble_artifact_sha256(baseline) != config.get("baseline", {}).get("artifact_sha256"):
        raise RuntimeError("structure-refiner baseline artifact hash mismatch")
    return config


def ensemble_artifact_sha256(checkpoint_dir: str | Path) -> str:
    from .local_fusion_runtime import ensemble_artifact_sha256 as baseline_sha256

    root = Path(checkpoint_dir).resolve()
    config = load_ensemble_config(root)
    digest = hashlib.sha256()
    digest.update(_sha256(root / "ensemble_config.json").encode())
    digest.update(baseline_sha256(root / "baseline").encode())
    for name, filenames in FILES.items():
        directory = _component(root, config, name)
        for filename in filenames:
            digest.update(name.encode() + b"/" + filename.encode() + b"\0")
            digest.update(_sha256(directory / filename).encode())
    return digest.hexdigest()


class Track2StructureGatedLocalFusion:
    def __init__(self, checkpoint_dir: str | Path, device: str = "cuda") -> None:
        import torch

        from .autoregressive_structure_unet_runtime import Track2AutoregressiveStructureUNet
        from .local_fusion_runtime import Track2LocalMotionTextureFusion
        from .structure_gated_refiner import StructureGatedHighFrequencyRefiner

        self.root = Path(checkpoint_dir).resolve()
        self.config = load_ensemble_config(self.root)
        self.device, self.torch = torch.device(device), torch
        self.baseline = Track2LocalMotionTextureFusion(self.root / "baseline", device)
        self.structure_parent = Track2AutoregressiveStructureUNet(_component(self.root, self.config, "structure_parent"), device)
        refiner_dir = _component(self.root, self.config, "refiner")
        model_config = np.load(refiner_dir / "structure_refiner_config.npz", allow_pickle=False)
        self.model = StructureGatedHighFrequencyRefiner(int(model_config["base_channels"]), float(model_config["residual_scale"])).to(self.device).eval()
        state = torch.load(refiner_dir / "model.pt", map_location="cpu", weights_only=True)
        if state.get("format") != "track2-structure-gated-high-frequency-refiner-v1":
            raise RuntimeError("unsupported structure-gated refiner model")
        self.model.load_state_dict(state["state_dict"], strict=True)
        normalization = np.load(refiner_dir / "action_normalization.npz", allow_pickle=False)
        self.mean = np.asarray(normalization["mean"], dtype=np.float32)
        self.std = np.asarray(normalization["std"], dtype=np.float32)

    @property
    def metadata(self) -> dict[str, object]:
        return {"format": FORMAT, "artifact_sha256": ensemble_artifact_sha256(self.root)}

    def predict(self, context_frames, history_actions, future_actions, seed: int, instruction: str | None):
        if context_frames.shape != (CONTEXT_FRAMES, 256, 256, 3) or context_frames.dtype != np.uint8:
            raise ValueError("context_frames must be [5,256,256,3] uint8")
        if history_actions.shape != (CONTEXT_ACTIONS, ACTION_DIM) or future_actions.shape != (PREDICTION_FRAMES, ACTION_DIM):
            raise ValueError("actions must be [4,14] history and [8,14] future")
        baseline = self.baseline.predict(context_frames, history_actions, future_actions, seed, instruction)
        structure, masks = self.structure_parent.predict_with_structure(context_frames, history_actions, future_actions, seed, instruction)
        torch = self.torch
        context = torch.from_numpy(context_frames[-1:].copy()).permute(0, 3, 1, 2).unsqueeze(0).to(self.device).float().div(255.0)
        baseline_tensor = torch.from_numpy(baseline.copy()).permute(0, 3, 1, 2).unsqueeze(0).to(self.device).float().div(255.0)
        structure_tensor = torch.from_numpy(structure.copy()).permute(0, 3, 1, 2).unsqueeze(0).to(self.device).float().div(255.0)
        # Training/evaluation caches store masks as uint8; preserve that exact
        # deployment contract so packaged and cached inference are bit-aligned.
        quantized_masks = np.rint(masks * 255.0).clip(0, 255).astype(np.uint8)
        mask_tensor = torch.from_numpy(quantized_masks).unsqueeze(0).to(self.device).float().div(255.0)
        actions = torch.from_numpy(np.ascontiguousarray((future_actions - self.mean) / self.std)).unsqueeze(0).to(self.device).float()
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            prediction, _, _, _ = self.model(context, baseline_tensor, structure_tensor, mask_tensor, actions)
        return prediction.clamp(0, 1).mul(255).round().to(torch.uint8).squeeze(0).permute(0, 2, 3, 1).cpu().numpy().copy()
