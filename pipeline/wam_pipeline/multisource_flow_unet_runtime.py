"""Runtime loader for the five-source flow-supervised Track 2 model."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .multisource_flow_unet import MultiSourceActionFlowUNet
from .profile import ACTION_DIM, CONTEXT_FRAMES, PREDICTION_FRAMES


class Track2MultiSourceFlowUNet:
    def __init__(self, checkpoint_dir: str | Path, device: str = "cuda") -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device)
        root = Path(checkpoint_dir)
        config = np.load(root / "track2_multisource_flow_unet_config.npz", allow_pickle=False)
        expected = (CONTEXT_FRAMES, ACTION_DIM, PREDICTION_FRAMES)
        actual = (int(config["context_frames"]), int(config["action_dim"]), int(config["prediction_frames"]))
        if actual != expected:
            raise RuntimeError("multi-source flow U-Net checkpoint does not match the official Track 2 profile")
        normalization = np.load(root / "action_normalization.npz", allow_pickle=False)
        self.mean = np.asarray(normalization["mean"], dtype=np.float32)
        self.std = np.asarray(normalization["std"], dtype=np.float32)
        state = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
        if state.get("format") != "track2-multisource-flow-unet-v1":
            raise RuntimeError("unsupported multi-source flow U-Net checkpoint format")
        self.model = MultiSourceActionFlowUNet(int(config["base_channels"])).to(self.device).eval()
        self.model.load_state_dict(state["state_dict"], strict=True)

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        if context_frames.shape != (CONTEXT_FRAMES, 256, 256, 3) or context_frames.dtype != np.uint8:
            raise ValueError("context frames must be [5,256,256,3] uint8")
        if history_actions.shape != (4, ACTION_DIM) or future_actions.shape != (8, ACTION_DIM):
            raise ValueError("actions must be [4,14] history and [8,14] future")
        torch = self.torch
        torch.manual_seed(int(seed))
        context = torch.from_numpy(np.ascontiguousarray(context_frames)).permute(0, 3, 1, 2).float().div(255.0)
        context = context.unsqueeze(0).to(self.device)
        actions = np.concatenate((history_actions, future_actions))
        actions = torch.from_numpy(np.ascontiguousarray((actions - self.mean) / self.std)).unsqueeze(0).to(self.device)
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            prediction = self.model(context, actions.float()).clamp(0.0, 1.0).mul(255).round().to(torch.uint8)
        return prediction.squeeze(0).permute(0, 2, 3, 1).cpu().numpy().copy()

    def predict_batch(self, context_frames, history_actions, future_actions, seeds, instructions):
        del seeds, instructions
        batch = int(context_frames.shape[0])
        if context_frames.shape != (batch, CONTEXT_FRAMES, 256, 256, 3) or context_frames.dtype != np.uint8:
            raise ValueError("context frames must be [B,5,256,256,3] uint8")
        if history_actions.shape != (batch, 4, ACTION_DIM) or future_actions.shape != (batch, 8, ACTION_DIM):
            raise ValueError("actions must be [B,4,14] history and [B,8,14] future")
        torch = self.torch
        context = torch.from_numpy(np.ascontiguousarray(context_frames)).permute(0, 1, 4, 2, 3).float().div(255.0).to(self.device)
        actions = np.concatenate((history_actions, future_actions), axis=1)
        actions = torch.from_numpy(np.ascontiguousarray((actions - self.mean) / self.std)).to(self.device)
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            prediction = self.model(context, actions.float()).clamp(0.0, 1.0).mul(255).round().to(torch.uint8)
        return prediction.permute(0, 1, 3, 4, 2).cpu().numpy().copy()
