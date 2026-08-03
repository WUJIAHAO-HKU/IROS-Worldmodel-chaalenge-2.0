"""Per-future-step action-conditioned U-Net for deterministic Track 2 video prediction."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn

from .residual_unet import ConvBlock, DownBlock, UpBlock


class TemporalActionUNet(nn.Module):
    """Decode each future image with its own action-prefix embedding."""

    context_frames = 5
    history_actions = 4
    prediction_frames = 8
    action_dim = 14

    def __init__(self) -> None:
        super().__init__()
        base = 64
        self.enc0 = ConvBlock(self.context_frames * 3 + 2, base)
        self.enc1 = DownBlock(base, 128)
        self.enc2 = DownBlock(128, 192)
        self.enc3 = DownBlock(192, 256)
        self.enc4 = DownBlock(256, 320)
        self.action_gru = nn.GRU(self.action_dim, 320, batch_first=True)
        self.action_norm = nn.LayerNorm(320)
        self.bottleneck_scale = nn.Linear(320, 320)
        self.bottleneck_shift = nn.Linear(320, 320)
        self.middle = ConvBlock(320, 320)
        self.up3 = UpBlock(320, 256, 256)
        self.up2 = UpBlock(256, 192, 192)
        self.up1 = UpBlock(192, 128, 128)
        self.up0 = UpBlock(128, base, base)
        self.final_condition = nn.Sequential(nn.Linear(320, base), nn.SiLU(), nn.Linear(base, base))
        self.output = nn.Conv2d(base, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

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

    def forward(self, context_frames: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = context_frames.shape
        if frames != self.context_frames or channels != 3:
            raise ValueError("context_frames must be [B,5,3,H,W]")
        if actions.shape != (batch, self.history_actions + self.prediction_frames, self.action_dim):
            raise ValueError("actions must be [B,12,14]")
        encoded_input = torch.cat(
            [context_frames.flatten(1, 2), self._coordinates(batch, height, width, context_frames.device, context_frames.dtype)], dim=1
        )
        e0 = self.enc0(encoded_input)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        action_states, _ = self.action_gru(actions)
        # State after each predicted action carries its complete action prefix.
        action_states = self.action_norm(action_states[:, self.history_actions :])
        steps = self.prediction_frames
        condition = action_states.flatten(0, 1)
        bottleneck = self._repeat(e4, steps)
        scale = self.bottleneck_scale(condition).unsqueeze(-1).unsqueeze(-1)
        shift = self.bottleneck_shift(condition).unsqueeze(-1).unsqueeze(-1)
        decoded = self.middle(bottleneck * (1.0 + scale) + shift)
        decoded = self.up3(decoded, self._repeat(e3, steps))
        decoded = self.up2(decoded, self._repeat(e2, steps))
        decoded = self.up1(decoded, self._repeat(e1, steps))
        decoded = self.up0(decoded, self._repeat(e0, steps))
        decoded = decoded + self.final_condition(condition).unsqueeze(-1).unsqueeze(-1)
        residual = self.output(decoded).reshape(batch, steps, 3, height, width)
        return context_frames[:, -1:] + residual
