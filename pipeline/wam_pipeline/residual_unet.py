"""A compact action-conditioned residual video model for Track 2."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as functional


class ConvBlock(nn.Module):
    """Two normalized convolutions used at one image scale."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.block = ConvBlock(out_channels, out_channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(self.downsample(value))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = functional.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([value, skip], dim=1))


class ActionConditionedResidualUNet(nn.Module):
    """Predict eight RGB residuals from five observations and 4+8 actions."""

    context_frames = 5
    history_actions = 4
    prediction_frames = 8
    action_dim = 14
    base_channels = 48

    def __init__(self) -> None:
        super().__init__()
        base = self.base_channels
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
        self.output = nn.Conv2d(base, self.prediction_frames * 3, kernel_size=3, padding=1)

    def forward(self, context_frames: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Return unclamped RGB predictions in [B,8,3,H,W]."""
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
        middle = self.middle(torch.cat([e4, action.expand(-1, -1, *e4.shape[-2:])], dim=1))
        decoded = self.up3(middle, e3)
        decoded = self.up2(decoded, e2)
        decoded = self.up1(decoded, e1)
        decoded = self.up0(decoded, e0)
        residual = self.output(decoded).reshape(batch, self.prediction_frames, 3, height, width)
        return context_frames[:, -1:].expand(-1, self.prediction_frames, -1, -1, -1) + residual
