"""Small bounded residual refiner applied after the complete frozen V15 model."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def block(channels: int) -> nn.Sequential:
    groups = min(8, channels)
    return nn.Sequential(
        nn.Conv2d(channels, channels, 3, padding=1),
        nn.GroupNorm(groups, channels),
        nn.SiLU(),
        nn.Conv2d(channels, channels, 3, padding=1),
        nn.GroupNorm(groups, channels),
        nn.SiLU(),
    )


class PostV15ResidualUNet(nn.Module):
    def __init__(
        self,
        base_channels: int = 16,
        maximum_residual_255: float = 8.0,
        initial_gate_probability: float = 0.05,
    ) -> None:
        super().__init__()
        if maximum_residual_255 <= 0:
            raise ValueError("maximum residual must be positive")
        if not 0 < initial_gate_probability < 1:
            raise ValueError("initial gate probability must be in (0, 1)")
        self.base_channels = int(base_channels)
        self.maximum_residual = float(maximum_residual_255) / 255.0
        width = self.base_channels
        self.input = nn.Conv2d(9, width, 3, padding=1)
        self.enc0 = block(width)
        self.down1 = nn.Conv2d(width, 2 * width, 4, stride=2, padding=1)
        self.enc1 = block(2 * width)
        self.down2 = nn.Conv2d(2 * width, 4 * width, 4, stride=2, padding=1)
        self.bottleneck = block(4 * width)
        self.condition = nn.Sequential(
            nn.Linear(10, 4 * width), nn.SiLU(), nn.Linear(4 * width, 4 * width)
        )
        self.up1 = nn.ConvTranspose2d(4 * width, 2 * width, 4, stride=2, padding=1)
        self.dec1 = block(4 * width)
        self.reduce1 = nn.Conv2d(4 * width, 2 * width, 1)
        self.up0 = nn.ConvTranspose2d(2 * width, width, 4, stride=2, padding=1)
        self.dec0 = block(2 * width)
        self.reduce0 = nn.Conv2d(2 * width, width, 1)
        self.output = nn.Conv2d(width, 4, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.output.bias.data[3] = math.log(
            initial_gate_probability / (1 - initial_gate_probability)
        )

    def forward(
        self,
        baseline: torch.Tensor,
        context_last: torch.Tensor,
        active_action: torch.Tensor,
        horizon: torch.Tensor,
        arm_right: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        difference = baseline - context_last
        x = torch.cat((baseline, context_last, difference), 1)
        x0 = self.enc0(self.input(x))
        x1 = self.enc1(self.down1(x0))
        x2 = self.bottleneck(self.down2(x1))
        arm = F.one_hot(arm_right.long(), 2).to(active_action.dtype)
        condition = torch.cat((active_action, horizon[:, None], arm), 1)
        x2 = x2 + self.condition(condition)[:, :, None, None]
        y1 = self.up1(x2)
        y1 = self.reduce1(self.dec1(torch.cat((y1, x1), 1)))
        y0 = self.up0(y1)
        y0 = self.reduce0(self.dec0(torch.cat((y0, x0), 1)))
        raw = self.output(y0)
        residual = torch.tanh(raw[:, :3]) * self.maximum_residual
        gate = torch.sigmoid(raw[:, 3:4])
        output = (baseline + gate * residual).clamp(0, 1)
        return output, {"residual": residual, "gate": gate}


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
