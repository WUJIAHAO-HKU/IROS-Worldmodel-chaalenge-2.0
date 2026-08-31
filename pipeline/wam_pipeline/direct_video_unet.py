"""High-capacity direct eight-frame predictor for the Track 2 RGB contract."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn

from .residual_unet import ConvBlock, DownBlock, UpBlock


class DirectActionVideoUNet(nn.Module):
    """Decode each future horizon directly so errors do not feed later frames."""

    context_frames = 5
    history_actions = 4
    prediction_frames = 8
    action_dim = 14

    def __init__(self) -> None:
        super().__init__()
        base = 80
        # Five images, two recent motion differences, and pixel coordinates.
        self.enc0 = ConvBlock(self.context_frames * 3 + 3 + 3 + 2, base)
        self.enc1 = DownBlock(base, 160)
        self.enc2 = DownBlock(160, 240)
        self.enc3 = DownBlock(240, 320)
        self.enc4 = DownBlock(320, 384)
        self.action_gru = nn.GRU(self.action_dim, 384, batch_first=True)
        self.action_norm = nn.LayerNorm(384)
        self.middle = ConvBlock(384, 384)
        self.up3 = UpBlock(384, 320, 320)
        self.up2 = UpBlock(320, 240, 240)
        self.up1 = UpBlock(240, 160, 160)
        self.up0 = UpBlock(160, base, base)
        self.film = nn.ModuleList(nn.Linear(384, channels * 2) for channels in (384, 320, 240, 160, base))
        self.output = nn.Conv2d(base, 3, kernel_size=3, padding=1)
        # Start from the stable copy-last-frame predictor and learn corrections.
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

    @staticmethod
    def _modulate(value: torch.Tensor, condition: torch.Tensor, projection: nn.Linear) -> torch.Tensor:
        scale, shift = projection(condition).chunk(2, dim=1)
        return value * (1.0 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)

    def forward(self, context_frames: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = context_frames.shape
        if (frames, channels) != (self.context_frames, 3):
            raise ValueError("context_frames must be [B,5,3,H,W]")
        if actions.shape != (batch, self.history_actions + self.prediction_frames, self.action_dim):
            raise ValueError("actions must be [B,12,14]")

        visual = torch.cat(
            (
                context_frames.flatten(1, 2),
                context_frames[:, -1] - context_frames[:, -2],
                context_frames[:, -1] - context_frames[:, 0],
                self._coordinates(batch, height, width, context_frames.device, context_frames.dtype),
            ),
            dim=1,
        )
        e0 = self.enc0(visual)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        action_states, _ = self.action_gru(actions)
        condition = self.action_norm(action_states[:, self.history_actions :]).flatten(0, 1)
        steps = self.prediction_frames

        decoded = self._modulate(self.middle(self._repeat(e4, steps)), condition, self.film[0])
        decoded = self._modulate(self.up3(decoded, self._repeat(e3, steps)), condition, self.film[1])
        decoded = self._modulate(self.up2(decoded, self._repeat(e2, steps)), condition, self.film[2])
        decoded = self._modulate(self.up1(decoded, self._repeat(e1, steps)), condition, self.film[3])
        decoded = self._modulate(self.up0(decoded, self._repeat(e0, steps)), condition, self.film[4])
        residual = self.output(decoded).reshape(batch, steps, 3, height, width)
        return context_frames[:, -1:] + residual
