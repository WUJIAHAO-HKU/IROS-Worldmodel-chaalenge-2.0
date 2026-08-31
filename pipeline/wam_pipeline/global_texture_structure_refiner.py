"""High-capacity full-frame restoration on top of an accurate motion parent."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .residual_unet import ConvBlock, DownBlock, UpBlock


class GlobalTextureStructureRefiner(nn.Module):
    context_frames = 5
    prediction_frames = 8
    action_dim = 14

    def __init__(self, base_channels: int = 32, residual_scale: float = 0.5) -> None:
        super().__init__()
        self.base_channels = int(base_channels)
        self.residual_scale = float(residual_scale)
        base = self.base_channels
        # All context RGB, parent RGB, displacement, source/parent high-pass,
        # two coordinates, motion magnitude, and horizon.
        input_channels = 15 + 3 + 3 + 3 + 3 + 2 + 1 + 1
        self.enc0 = ConvBlock(input_channels, base)
        self.enc1 = DownBlock(base, base * 2)
        self.enc2 = DownBlock(base * 2, base * 3)
        self.enc3 = DownBlock(base * 3, base * 4)
        self.enc4 = DownBlock(base * 4, base * 6)
        self.middle = ConvBlock(base * 6, base * 6)
        self.action_gru = nn.GRU(self.action_dim, base * 6, batch_first=True)
        self.action_film = nn.Linear(base * 6, base * 12)
        self.up3 = UpBlock(base * 6, base * 4, base * 4)
        self.up2 = UpBlock(base * 4, base * 3, base * 3)
        self.up1 = UpBlock(base * 3, base * 2, base * 2)
        self.up0 = UpBlock(base * 2, base, base)
        self.output = nn.Conv2d(base, 4, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def highpass(value: torch.Tensor) -> torch.Tensor:
        return value - F.avg_pool2d(value, 5, stride=1, padding=2, count_include_pad=False)

    @staticmethod
    def coordinates(batch: int, height: int, width: int, device, dtype) -> torch.Tensor:
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=device, dtype=dtype),
            torch.linspace(-1, 1, width, device=device, dtype=dtype), indexing="ij",
        )
        return torch.stack((x, y)).unsqueeze(0).expand(batch, -1, -1, -1)

    def forward(self, context: torch.Tensor, parent: torch.Tensor, actions: torch.Tensor):
        batch, frames, channels, height, width = context.shape
        if (frames, channels) != (5, 3) or parent.shape != (batch, 8, 3, height, width):
            raise ValueError("expected context [B,5,3,H,W] and parent [B,8,3,H,W]")
        if actions.shape != (batch, 12, 14):
            raise ValueError("actions must be [B,12,14]")
        steps = parent.shape[1]
        source = context.flatten(1, 2)[:, None].expand(-1, steps, -1, -1, -1)
        anchor = context[:, -1:,].expand(-1, steps, -1, -1, -1)
        displacement = parent - anchor
        parent_hp = self.highpass(parent.flatten(0, 1)).unflatten(0, (batch, steps))
        anchor_hp = self.highpass(anchor.flatten(0, 1)).unflatten(0, (batch, steps))
        coords = self.coordinates(batch, height, width, parent.device, parent.dtype)[:, None].expand(-1, steps, -1, -1, -1)
        magnitude = displacement.abs().mean(2, keepdim=True)
        horizon = torch.linspace(1 / 8, 1, steps, device=parent.device, dtype=parent.dtype).view(1, steps, 1, 1, 1).expand(batch, -1, -1, height, width)
        visual = torch.cat((source, parent, displacement, parent_hp, anchor_hp, coords, magnitude, horizon), 2).flatten(0, 1)
        e0 = self.enc0(visual); e1 = self.enc1(e0); e2 = self.enc2(e1); e3 = self.enc3(e2); e4 = self.enc4(e3)
        states, _ = self.action_gru(actions)
        scale, shift = self.action_film(states[:, 4:].flatten(0, 1)).chunk(2, 1)
        middle = self.middle(e4) * (1 + scale[..., None, None]) + shift[..., None, None]
        decoded = self.up3(middle, e3); decoded = self.up2(decoded, e2); decoded = self.up1(decoded, e1); decoded = self.up0(decoded, e0)
        output = self.output(decoded).unflatten(0, (batch, steps))
        gate = torch.sigmoid(output[:, :, :1])
        correction = gate * self.residual_scale * torch.tanh(output[:, :, 1:])
        return parent + correction, correction, gate
