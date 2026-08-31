"""Native-resolution direct predictor used only for the v25 fit diagnostic."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), norm(cout), nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1), norm(cout), nn.SiLU(),
    )


class DirectOverfitDiagnosticV250(nn.Module):
    """Predict RGB directly from all observations, parent and full action plan."""

    visual_channels = 24  # parent, previous parent, five observations, last-frame high pass

    def __init__(self, channels: int = 32, maximum_residual: float = .75) -> None:
        super().__init__(); c = channels
        self.channels = int(channels); self.maximum_residual = float(maximum_residual)
        self.action = nn.Sequential(nn.Linear(58, 128), nn.SiLU(), nn.Linear(128, 8 * 16))
        self.enc0 = block(self.visual_channels + 16, c)
        self.down1 = nn.Conv2d(c, 2 * c, 4, 2, 1); self.enc1 = block(2 * c, 2 * c)
        self.down2 = nn.Conv2d(2 * c, 4 * c, 4, 2, 1); self.enc2 = block(4 * c, 4 * c)
        self.down3 = nn.Conv2d(4 * c, 6 * c, 4, 2, 1)
        self.middle = nn.Sequential(block(6 * c, 6 * c), block(6 * c, 6 * c))
        self.dec2 = block(10 * c, 4 * c); self.dec1 = block(6 * c, 2 * c)
        self.dec0 = block(3 * c, c); self.head = nn.Conv2d(c, 3, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    @staticmethod
    def highpass(value: torch.Tensor) -> torch.Tensor:
        return value - F.avg_pool2d(value, 5, 1, 2, count_include_pad=False)

    def forward(self, base: torch.Tensor, context: torch.Tensor, actions: torch.Tensor,
                arm: torch.Tensor) -> torch.Tensor:
        batch, time, channels, height, width = base.shape
        if context.shape != (batch, 5, 3, height, width) or actions.shape != (batch, 8, 7):
            raise ValueError("base/context/action rollout shapes")
        arm_one_hot = F.one_hot(arm.long(), 2).to(base.dtype)
        condition = self.action(torch.cat((actions.flatten(1), arm_one_hot), 1)).reshape(batch, 8, 16)
        context_flat = context.flatten(1, 2)[:, None].expand(-1, time, -1, -1, -1)
        previous = torch.cat((context[:, -1:,], base[:, :-1]), 1)
        last_high = self.highpass(context[:, -1])[:, None].expand(-1, time, -1, -1, -1)
        visual = torch.cat((base, previous, context_flat, last_high), 2)
        condition = condition[:, :, :, None, None].expand(-1, -1, -1, height, width)
        value = torch.cat((visual, condition), 2).flatten(0, 1)
        e0 = self.enc0(value); e1 = self.enc1(self.down1(e0)); e2 = self.enc2(self.down2(e1))
        middle = self.middle(self.down3(e2))
        d2 = F.interpolate(middle, e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat((d2, e2), 1))
        d1 = F.interpolate(d2, e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat((d1, e1), 1))
        d0 = F.interpolate(d1, e0.shape[-2:], mode="bilinear", align_corners=False)
        residual = self.maximum_residual * torch.tanh(self.head(self.dec0(torch.cat((d0, e0), 1))))
        return (base.flatten(0, 1) + residual).clamp(0, 1).unflatten(0, (batch, time))


def parameter_count(channels: int = 32) -> int:
    return sum(parameter.numel() for parameter in DirectOverfitDiagnosticV250(channels).parameters())
