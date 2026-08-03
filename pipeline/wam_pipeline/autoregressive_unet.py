"""One-step action-conditioned model used recursively for Track 2 rollouts."""

from __future__ import annotations

import torch
from torch import nn

from .residual_unet import ConvBlock, DownBlock, UpBlock


class OneStepActionUNet(nn.Module):
    """Predict frame t+1 from five RGB frames and the five preceding actions."""

    context_frames = 5
    conditioned_actions = 5
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
            nn.Linear(self.conditioned_actions * self.action_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
        )
        self.middle = ConvBlock(512, 256)
        self.up3 = UpBlock(256, 192, 192)
        self.up2 = UpBlock(192, 144, 144)
        self.up1 = UpBlock(144, 96, 96)
        self.up0 = UpBlock(96, base, base)
        self.output = nn.Conv2d(base, 3, kernel_size=3, padding=1)

    def forward(self, context_frames: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = context_frames.shape
        if (frames, channels) != (self.context_frames, 3):
            raise ValueError("context_frames must be [B,5,3,H,W]")
        if actions.shape != (batch, self.conditioned_actions, self.action_dim):
            raise ValueError("actions must be [B,5,14]")
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
        return context_frames[:, -1] + self.output(decoded)
