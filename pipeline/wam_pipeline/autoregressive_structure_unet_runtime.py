"""Runtime for the auxiliary-structure autoregressive Track 2 parent."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .autoregressive_structure_unet import OneStepActionSeparatedStructureUNet, OneStepActionStructureUNet
from .profile import ACTION_DIM, CONTEXT_FRAMES, PREDICTION_FRAMES


class Track2AutoregressiveStructureUNet:
    def __init__(self, checkpoint_dir: str | Path, device: str = "cuda") -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device)
        root = Path(checkpoint_dir)
        config = np.load(root / "track2_autoregressive_structure_unet_config.npz", allow_pickle=False)
        if (int(config["context_frames"]), int(config["action_dim"]), int(config["prediction_frames"])) != (CONTEXT_FRAMES, ACTION_DIM, PREDICTION_FRAMES):
            raise RuntimeError("structure U-Net checkpoint does not match Track 2")
        normalization = np.load(root / "action_normalization.npz", allow_pickle=False)
        self.mean = np.asarray(normalization["mean"], np.float32)
        self.std = np.asarray(normalization["std"], np.float32)
        state = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
        if state.get("format") not in {"track2-autoregressive-structure-unet-v1", "track2-autoregressive-structure-unet-v2", "track2-autoregressive-structure-unet-v3"}:
            raise RuntimeError("unsupported structure U-Net checkpoint")
        model_class = OneStepActionSeparatedStructureUNet if state.get("format") == "track2-autoregressive-structure-unet-v3" else OneStepActionStructureUNet
        self.model = model_class().to(self.device).eval()
        incompatible = self.model.load_state_dict(state["state_dict"], strict=False)
        if incompatible.unexpected_keys or set(incompatible.missing_keys) - {"structure_strength"}:
            raise RuntimeError("structure U-Net state does not match runtime")

    def predict(self, context_frames, history_actions, future_actions, seed: int, instruction: str | None):
        prediction, _ = self.predict_with_structure(context_frames, history_actions, future_actions, seed, instruction)
        return prediction

    def predict_with_structure(self, context_frames, history_actions, future_actions, seed: int, instruction: str | None):
        """Return the native RGB rollout and the sigmoid structure probability maps."""
        del instruction
        if context_frames.shape != (CONTEXT_FRAMES, 256, 256, 3) or context_frames.dtype != np.uint8:
            raise ValueError("context_frames must be [5,256,256,3] uint8")
        if history_actions.shape != (4, ACTION_DIM) or future_actions.shape != (8, ACTION_DIM):
            raise ValueError("actions must be [4,14] history and [8,14] future")
        torch = self.torch
        torch.manual_seed(int(seed))
        context = torch.from_numpy(np.ascontiguousarray(context_frames)).permute(0, 3, 1, 2).float().div(255.0).unsqueeze(0).to(self.device)
        history = torch.from_numpy(np.ascontiguousarray((history_actions - self.mean) / self.std)).unsqueeze(0).to(self.device)
        future = torch.from_numpy(np.ascontiguousarray((future_actions - self.mean) / self.std)).unsqueeze(0).to(self.device)
        predictions, structures = [], []
        with torch.inference_mode():
            for action in future.unbind(dim=1):
                prediction, structure = self.model(context, torch.cat((history, action[:, None]), dim=1), return_structure=True)
                prediction = prediction.clamp(0, 1)
                predictions.append(prediction)
                structures.append(structure.sigmoid())
                recurrent = (prediction + self.model.structure_correction(structure)).clamp(0, 1)
                context = torch.cat((context[:, 1:], recurrent[:, None]), dim=1)
                history = torch.cat((history[:, 1:], action[:, None]), dim=1)
        rgb = torch.stack(predictions, dim=1).mul(255).round().byte().squeeze(0).permute(0, 2, 3, 1).cpu().numpy().copy()
        masks = torch.stack(structures, dim=1).float().squeeze(0).cpu().numpy().copy()
        return rgb, masks
