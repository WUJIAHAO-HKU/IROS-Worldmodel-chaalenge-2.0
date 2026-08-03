"""Motion-warping Track 2 predictor that preserves appearance from the last frame."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn

from .residual_unet import ConvBlock, DownBlock, UpBlock


class ActionConditionedFlowResidualUNet(nn.Module):
    """Predict future RGB frames as a warped observation plus an RGB correction."""

    context_frames = 5
    history_actions = 4
    prediction_frames = 8
    action_dim = 14
    max_flow_pixels = 48.0

    def __init__(self) -> None:
        super().__init__()
        base = 48
        self.enc0 = ConvBlock(self.context_frames * 3, base)
        self.enc1 = DownBlock(base, 96)
        self.enc2 = DownBlock(96, 144)
        self.enc3 = DownBlock(144, 192)
        self.enc4 = DownBlock(192, 256)
        self.action_embedding = nn.Sequential(
            nn.Linear((self.history_actions + self.prediction_frames) * self.action_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
        )
        self.middle = ConvBlock(512, 256)
        self.up3 = UpBlock(256, 192, 192)
        self.up2 = UpBlock(192, 144, 144)
        self.up1 = UpBlock(144, 96, 96)
        self.up0 = UpBlock(96, base, base)
        # Each future frame has dx/dy flow, RGB correction, and blend logit.
        self.output = nn.Conv2d(base, self.prediction_frames * 6, kernel_size=3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        with torch.no_grad():
            self.output.bias.view(self.prediction_frames, 6)[:, 5].fill_(2.0)

    @staticmethod
    def _warp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Sample image with pixel-unit flow maps shaped [B,T,2,H,W]."""
        batch, steps, _, height, width = flow.shape
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=image.device, dtype=image.dtype),
            torch.linspace(-1.0, 1.0, width, device=image.device, dtype=image.dtype),
            indexing="ij",
        )
        base = torch.stack((x, y), dim=-1).view(1, 1, height, width, 2)
        scale = torch.tensor((2.0 / max(width - 1, 1), 2.0 / max(height - 1, 1)), device=image.device, dtype=image.dtype)
        grid = base + flow.permute(0, 1, 3, 4, 2) * scale
        source = image[:, None].expand(-1, steps, -1, -1, -1).reshape(batch * steps, 3, height, width)
        return functional.grid_sample(
            source,
            grid.reshape(batch * steps, height, width, 2),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).reshape(batch, steps, 3, height, width)

    def forward(self, context_frames: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = context_frames.shape
        if frames != self.context_frames or channels != 3:
            raise ValueError("context_frames must be [B,5,3,H,W]")
        if actions.shape != (batch, self.history_actions + self.prediction_frames, self.action_dim):
            raise ValueError("actions must be [B,12,14]")
        e0 = self.enc0(context_frames.reshape(batch, frames * channels, height, width))
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        action = self.action_embedding(actions.flatten(1)).unsqueeze(-1).unsqueeze(-1)
        decoded = self.middle(torch.cat([e4, action.expand(-1, -1, *e4.shape[-2:])], dim=1))
        decoded = self.up3(decoded, e3)
        decoded = self.up2(decoded, e2)
        decoded = self.up1(decoded, e1)
        decoded = self.up0(decoded, e0)
        output = self.output(decoded).reshape(batch, self.prediction_frames, 6, height, width)
        flow = torch.tanh(output[:, :, :2]) * self.max_flow_pixels
        correction = torch.tanh(output[:, :, 2:5]) * 0.5
        blend = torch.sigmoid(output[:, :, 5:6])
        warped = self._warp(context_frames[:, -1], flow)
        return blend * warped + (1.0 - blend) * context_frames[:, -1:] + correction
