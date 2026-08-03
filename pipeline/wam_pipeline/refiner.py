"""A lightweight image refiner for frozen Track 2 rollout predictions."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(value + self.layers(value))


class Track2ImageRefiner(nn.Module):
    """Predict an RGB correction using a frozen predicted frame and last observation."""

    def __init__(self) -> None:
        super().__init__()
        channels = 64
        self.input = nn.Sequential(nn.Conv2d(6, channels, 3, padding=1), nn.GroupNorm(8, channels), nn.SiLU())
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(6)])
        self.output = nn.Conv2d(channels, 3, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, prediction: torch.Tensor, last_observation: torch.Tensor) -> torch.Tensor:
        return prediction + self.output(self.blocks(self.input(torch.cat([prediction, last_observation], dim=1))))


class ConditionalPatchDiscriminator(nn.Module):
    """Judge local realism of a next frame conditioned on the preceding image."""

    def __init__(self) -> None:
        super().__init__()
        channels = (64, 128, 256, 384)
        layers: list[nn.Module] = []
        previous = 6
        for index, current in enumerate(channels):
            layers.append(nn.Conv2d(previous, current, 4, stride=2 if index < 3 else 1, padding=1))
            if index:
                layers.append(nn.GroupNorm(8, current))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            previous = current
        layers.append(nn.Conv2d(previous, 1, 4, padding=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, previous_frame: torch.Tensor, next_frame: torch.Tensor) -> torch.Tensor:
        return self.layers(torch.cat([previous_frame, next_frame], dim=1))
