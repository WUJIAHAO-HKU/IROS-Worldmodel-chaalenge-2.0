"""Runtime loader for the Track 2 residual U-Net baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .profile import ACTION_DIM, CONTEXT_FRAMES, PREDICTION_FRAMES
from .residual_unet import ActionConditionedResidualUNet


class Track2ResidualUNet:
    def __init__(self, checkpoint_dir: str | Path, device: str = "cuda") -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device)
        root = Path(checkpoint_dir)
        config = np.load(root / "track2_residual_unet_config.npz", allow_pickle=False)
        if int(config["context_frames"]) != CONTEXT_FRAMES or int(config["action_dim"]) != ACTION_DIM:
            raise RuntimeError("residual U-Net checkpoint does not match the official Track 2 profile")
        if int(config["prediction_frames"]) != PREDICTION_FRAMES:
            raise RuntimeError("residual U-Net prediction length does not match the official Track 2 profile")
        self.resolution = int(config["working_resolution"])
        normalization = np.load(root / "action_normalization.npz", allow_pickle=False)
        self.mean = np.asarray(normalization["mean"], dtype=np.float32)
        self.std = np.asarray(normalization["std"], dtype=np.float32)
        if self.mean.shape != (ACTION_DIM,) or self.std.shape != (ACTION_DIM,) or np.any(self.std <= 0):
            raise RuntimeError("invalid residual U-Net action normalization")
        state = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
        if state.get("format") != "track2-residual-unet-v1":
            raise RuntimeError("unsupported residual U-Net checkpoint format")
        self.model = ActionConditionedResidualUNet().to(self.device).eval()
        self.model.load_state_dict(state["state_dict"], strict=True)

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        if context_frames.shape != (CONTEXT_FRAMES, 256, 256, 3) or context_frames.dtype != np.uint8:
            raise ValueError("context frames must be [5,256,256,3] uint8")
        if history_actions.shape != (4, ACTION_DIM) or future_actions.shape != (8, ACTION_DIM):
            raise ValueError("actions must be [4,14] history and [8,14] future")
        torch = self.torch
        torch.manual_seed(int(seed))
        context = torch.from_numpy(np.ascontiguousarray(context_frames)).permute(0, 3, 1, 2).float().div(255.0)
        context = torch.nn.functional.interpolate(context, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False)
        context = context.unsqueeze(0).to(self.device)
        actions = np.concatenate([history_actions, future_actions])
        actions = (actions - self.mean) / self.std
        action_tensor = torch.from_numpy(np.ascontiguousarray(actions)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            prediction = self.model(context, action_tensor).clamp(0.0, 1.0)
            prediction = torch.nn.functional.interpolate(
                prediction.flatten(0, 1), size=(256, 256), mode="bilinear", align_corners=False
            ).reshape(1, PREDICTION_FRAMES, 3, 256, 256)
            prediction = prediction.mul(255).round().to(torch.uint8)
        return prediction.squeeze(0).permute(0, 2, 3, 1).cpu().numpy().copy()
