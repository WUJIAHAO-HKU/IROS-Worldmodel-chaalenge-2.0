"""Action-conditioned pure flow renderer with motion-balanced supervision."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn

from .residual_unet import ConvBlock, DownBlock, UpBlock


class MotionWeightedFlowRenderer(nn.Module):
    """Predict low-resolution backward flow and transport observed RGB exactly."""

    context_frames = 5
    history_actions = 4
    prediction_frames = 8
    action_dim = 14

    def __init__(self, base_channels: int = 24, max_flow_128: float = 64.0) -> None:
        super().__init__()
        if base_channels % 8:
            raise ValueError("base_channels must be divisible by eight")
        self.base_channels = int(base_channels)
        self.max_flow_128 = float(max_flow_128)
        base = self.base_channels
        self.enc0 = ConvBlock(23, base)
        self.enc1 = DownBlock(base, base * 2)
        self.enc2 = DownBlock(base * 2, base * 3)
        self.enc3 = DownBlock(base * 3, base * 4)
        self.middle = ConvBlock(base * 4, base * 4)
        self.action_gru = nn.GRU(self.action_dim, base * 4, batch_first=True)
        self.action_norm = nn.LayerNorm(base * 4)
        self.film = nn.Linear(base * 4, base * 8)
        self.up2 = UpBlock(base * 4, base * 3, base * 3)
        self.up1 = UpBlock(base * 3, base * 2, base * 2)
        self.up0 = UpBlock(base * 2, base, base)
        self.output = nn.Conv2d(base, 2, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _coordinates(batch: int, height: int, width: int, device, dtype) -> torch.Tensor:
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=device, dtype=dtype),
            torch.linspace(-1, 1, width, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack((x, y)).unsqueeze(0).expand(batch, -1, -1, -1)

    @staticmethod
    def warp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        batch, steps, _, height, width = flow.shape
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=image.device, dtype=image.dtype),
            torch.linspace(-1, 1, width, device=image.device, dtype=image.dtype),
            indexing="ij",
        )
        base = torch.stack((x, y), dim=-1).view(1, 1, height, width, 2)
        scale = flow.new_tensor((2 / (width - 1), 2 / (height - 1)))
        grid = base + flow.permute(0, 1, 3, 4, 2) * scale
        source = image[:, None].expand(-1, steps, -1, -1, -1).flatten(0, 1)
        return functional.grid_sample(
            source, grid.flatten(0, 1), mode="bilinear", padding_mode="border", align_corners=True
        ).unflatten(0, (batch, steps))

    def predict_flow_128(self, context_128: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = context_128.shape
        if (frames, channels, height, width) != (5, 3, 128, 128):
            raise ValueError("context_128 must be [B,5,3,128,128]")
        if actions.shape != (batch, 12, 14):
            raise ValueError("actions must be [B,12,14]")
        encoded = torch.cat(
            (
                context_128.flatten(1, 2),
                context_128[:, -1] - context_128[:, -2],
                context_128[:, -1] - context_128[:, 0],
                self._coordinates(batch, height, width, context_128.device, context_128.dtype),
            ),
            dim=1,
        )
        e0, conditions = self.enc0(encoded), self.action_norm(self.action_gru(actions)[0][:, 4:]).flatten(0, 1)
        e1, e2 = self.enc1(e0), None
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        repeat = lambda value: value[:, None].expand(-1, 8, -1, -1, -1).flatten(0, 1)
        middle = self.middle(repeat(e3))
        scale, shift = self.film(conditions).chunk(2, dim=1)
        middle = middle * (1 + scale[..., None, None]) + shift[..., None, None]
        decoded = self.up2(middle, repeat(e2))
        decoded = self.up1(decoded, repeat(e1))
        decoded = self.up0(decoded, repeat(e0))
        return (torch.tanh(self.output(decoded)) * self.max_flow_128).unflatten(0, (batch, 8))

    def forward(self, context_256: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = context_256.shape[0]
        context_128 = functional.interpolate(context_256.flatten(0, 1), (128, 128), mode="bilinear", align_corners=False).unflatten(0, (batch, 5))
        flow_128 = self.predict_flow_128(context_128, actions)
        flow_256 = functional.interpolate(flow_128.flatten(0, 1), (256, 256), mode="bilinear", align_corners=True).unflatten(0, (batch, 8)) * 2.0
        return self.warp(context_256[:, -1], flow_256), flow_128
