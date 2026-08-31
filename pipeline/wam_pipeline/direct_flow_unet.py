"""Direct action-conditioned flow and residual decoder for Track 2."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn

from .residual_unet import ConvBlock, DownBlock, UpBlock


class DirectActionFlowUNet(nn.Module):
    """Generate target-to-source flows for all eight horizons in one pass."""

    context_frames = 5
    history_actions = 4
    prediction_frames = 8
    action_dim = 14
    max_flow_pixels = 128.0

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        if base_channels < 8 or base_channels % 8:
            raise ValueError("base_channels must be a positive multiple of 8")
        self.base_channels = int(base_channels)
        base = self.base_channels
        c1, c2, c3, c4 = base * 2, base * 3, base * 4, base * 5
        self.enc0 = ConvBlock(self.context_frames * 3 + 3 + 3 + 2, base)
        self.enc1 = DownBlock(base, c1)
        self.enc2 = DownBlock(c1, c2)
        self.enc3 = DownBlock(c2, c3)
        self.enc4 = DownBlock(c3, c4)
        self.action_gru = nn.GRU(self.action_dim, c4, batch_first=True)
        self.action_norm = nn.LayerNorm(c4)
        self.middle = ConvBlock(c4, c4)
        self.up3 = UpBlock(c4, c3, c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.up1 = UpBlock(c2, c1, c1)
        self.up0 = UpBlock(c1, base, base)
        self.film = nn.ModuleList(nn.Linear(c4, channels * 2) for channels in (c4, c3, c2, c1, base))
        # Each horizon produces target-to-source flow, RGB correction, blend.
        self.output = nn.Conv2d(base, 6, kernel_size=3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        with torch.no_grad():
            self.output.bias[5] = 2.0

    @staticmethod
    def _coordinates(batch: int, height: int, width: int, device, dtype) -> torch.Tensor:
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack((x, y)).unsqueeze(0).expand(batch, -1, -1, -1)

    @staticmethod
    def _repeat(value: torch.Tensor, steps: int) -> torch.Tensor:
        return value[:, None].expand(-1, steps, *value.shape[1:]).flatten(0, 1)

    @staticmethod
    def _modulate(value: torch.Tensor, condition: torch.Tensor, projection: nn.Linear) -> torch.Tensor:
        scale, shift = projection(condition).chunk(2, dim=1)
        return value * (1.0 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)

    @staticmethod
    def _warp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        batch, steps, _, height, width = flow.shape
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=image.device, dtype=image.dtype),
            torch.linspace(-1.0, 1.0, width, device=image.device, dtype=image.dtype),
            indexing="ij",
        )
        base = torch.stack((x, y), dim=-1).view(1, 1, height, width, 2)
        scale = torch.tensor((2.0 / (width - 1), 2.0 / (height - 1)), device=image.device, dtype=image.dtype)
        grid = base + flow.permute(0, 1, 3, 4, 2) * scale
        source = image[:, None].expand(-1, steps, -1, -1, -1).reshape(batch * steps, 3, height, width)
        return functional.grid_sample(source, grid.reshape(batch * steps, height, width, 2), mode="bilinear", padding_mode="border", align_corners=True).reshape(batch, steps, 3, height, width)

    def forward(self, context_frames: torch.Tensor, actions: torch.Tensor, return_flow: bool = False):
        batch, frames, channels, height, width = context_frames.shape
        if (frames, channels) != (self.context_frames, 3):
            raise ValueError("context_frames must be [B,5,3,H,W]")
        if actions.shape != (batch, self.history_actions + self.prediction_frames, self.action_dim):
            raise ValueError("actions must be [B,12,14]")
        encoded = torch.cat(
            (
                context_frames.flatten(1, 2),
                context_frames[:, -1] - context_frames[:, -2],
                context_frames[:, -1] - context_frames[:, 0],
                self._coordinates(batch, height, width, context_frames.device, context_frames.dtype),
            ),
            dim=1,
        )
        e0 = self.enc0(encoded)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        states, _ = self.action_gru(actions)
        condition = self.action_norm(states[:, self.history_actions :]).flatten(0, 1)
        steps = self.prediction_frames
        decoded = self._modulate(self.middle(self._repeat(e4, steps)), condition, self.film[0])
        decoded = self._modulate(self.up3(decoded, self._repeat(e3, steps)), condition, self.film[1])
        decoded = self._modulate(self.up2(decoded, self._repeat(e2, steps)), condition, self.film[2])
        decoded = self._modulate(self.up1(decoded, self._repeat(e1, steps)), condition, self.film[3])
        decoded = self._modulate(self.up0(decoded, self._repeat(e0, steps)), condition, self.film[4])
        output = self.output(decoded).reshape(batch, steps, 6, height, width)
        flow = torch.tanh(output[:, :, :2]) * self.max_flow_pixels
        correction = torch.tanh(output[:, :, 2:5]) * 0.35
        blend = torch.sigmoid(output[:, :, 5:6])
        warped = self._warp(context_frames[:, -1], flow)
        prediction = blend * warped + (1.0 - blend) * context_frames[:, -1:] + correction
        return (prediction, flow) if return_flow else prediction
