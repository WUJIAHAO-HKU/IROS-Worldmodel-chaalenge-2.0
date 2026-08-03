"""Runtime adapter that rolls a one-step Track 2 model forward eight times."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .autoregressive_unet import OneStepActionUNet
from .profile import ACTION_DIM, CONTEXT_FRAMES, PREDICTION_FRAMES


class Track2AutoregressiveUNet:
    def __init__(self, checkpoint_dir: str | Path, device: str = "cuda") -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device)
        root = Path(checkpoint_dir)
        config = np.load(root / "track2_autoregressive_unet_config.npz", allow_pickle=False)
        if (int(config["context_frames"]), int(config["action_dim"]), int(config["prediction_frames"])) != (CONTEXT_FRAMES, ACTION_DIM, PREDICTION_FRAMES):
            raise RuntimeError("autoregressive U-Net checkpoint does not match the official Track 2 profile")
        normalization = np.load(root / "action_normalization.npz", allow_pickle=False)
        self.mean = np.asarray(normalization["mean"], dtype=np.float32)
        self.std = np.asarray(normalization["std"], dtype=np.float32)
        state = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
        if state.get("format") != "track2-autoregressive-unet-v1":
            raise RuntimeError("unsupported autoregressive U-Net checkpoint format")
        self.model = OneStepActionUNet().to(self.device).eval()
        self.model.load_state_dict(state["state_dict"], strict=True)

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        if context_frames.shape != (CONTEXT_FRAMES, 256, 256, 3) or context_frames.dtype != np.uint8:
            raise ValueError("context frames must be [5,256,256,3] uint8")
        if history_actions.shape != (4, ACTION_DIM) or future_actions.shape != (8, ACTION_DIM):
            raise ValueError("actions must be [4,14] history and [8,14] future")
        torch = self.torch
        torch.manual_seed(int(seed))
        context = torch.from_numpy(np.ascontiguousarray(context_frames)).permute(0, 3, 1, 2).float().div(255.0).unsqueeze(0).to(self.device)
        history = torch.from_numpy(np.ascontiguousarray((history_actions - self.mean) / self.std)).unsqueeze(0).to(self.device)
        future = torch.from_numpy(np.ascontiguousarray((future_actions - self.mean) / self.std)).unsqueeze(0).to(self.device)
        predictions = []
        with torch.inference_mode():
            for action in future.unbind(dim=1):
                next_frame = self.model(context, torch.cat([history, action[:, None]], dim=1)).clamp(0.0, 1.0)
                predictions.append(next_frame)
                context = torch.cat([context[:, 1:], next_frame[:, None]], dim=1)
                history = torch.cat([history[:, 1:], action[:, None]], dim=1)
        prediction = torch.stack(predictions, dim=1).mul(255).round().to(torch.uint8)
        return prediction.squeeze(0).permute(0, 2, 3, 1).cpu().numpy().copy()
