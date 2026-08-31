"""High-capacity eight-frame residual refiner for the data-closed v18 experiment."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def block(cin: int, cout: int) -> nn.Sequential:
    groups = max(min(cout // 8, 16), 1)
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, padding=1), nn.GroupNorm(groups, cout), nn.SiLU(),
        nn.Conv3d(cout, cout, 3, padding=1), nn.GroupNorm(groups, cout), nn.SiLU(),
    )


class VideoDetailRefinerV180(nn.Module):
    """Jointly refine all eight frames while retaining a strict identity path."""

    input_channels = 22

    def __init__(self, base_channels: int = 32, maximum_residual: float = .50) -> None:
        super().__init__(); c = base_channels; self.maximum_residual = maximum_residual
        self.enc0 = block(self.input_channels, c)
        self.down1 = nn.Conv3d(c, 2 * c, (1, 4, 4), (1, 2, 2), (0, 1, 1))
        self.enc1 = block(2 * c, 2 * c)
        self.down2 = nn.Conv3d(2 * c, 4 * c, (1, 4, 4), (1, 2, 2), (0, 1, 1))
        self.enc2 = block(4 * c, 4 * c)
        self.down3 = nn.Conv3d(4 * c, 6 * c, (1, 4, 4), (1, 2, 2), (0, 1, 1))
        self.middle = nn.Sequential(block(6 * c, 6 * c), block(6 * c, 6 * c))
        self.action = nn.Sequential(nn.Linear(9, 4 * c), nn.SiLU(), nn.Linear(4 * c, 12 * c))
        self.dec2 = block(10 * c, 4 * c); self.dec1 = block(6 * c, 2 * c)
        self.dec0 = block(3 * c, c)
        self.head = nn.Conv3d(c, 4, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        # RGB residual weights are exactly zero, so a usable alpha prior still
        # starts pixel-identical to the parent while avoiding a saturated gate.
        with torch.no_grad(): self.head.bias[3] = -2.0

    def forward(self, visual: torch.Tensor, actions: torch.Tensor,
                arm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inputs are ``visual[B,22,T,H,W]`` and active actions ``[B,T,7]``."""
        if visual.ndim != 5 or visual.shape[1] != self.input_channels:
            raise ValueError(f"visual must be [B,22,T,H,W], got {tuple(visual.shape)}")
        batch, _, time = visual.shape[:3]
        horizon = torch.linspace(1 / time, 1, time, device=visual.device,
                                 dtype=visual.dtype)[None, :, None].expand(batch, -1, -1)
        arm_value = arm.to(visual.dtype).reshape(batch, 1, 1).expand(-1, time, -1)
        condition = self.action(torch.cat((actions, horizon, arm_value), -1))
        scale, bias = condition.chunk(2, -1)
        e0 = self.enc0(visual); e1 = self.enc1(self.down1(e0)); e2 = self.enc2(self.down2(e1))
        value = self.middle(self.down3(e2))
        scale = scale.permute(0, 2, 1)[:, :, :, None, None]
        bias = bias.permute(0, 2, 1)[:, :, :, None, None]
        value = value * (1 + .1 * torch.tanh(scale)) + .1 * bias
        value = F.interpolate(value, e2.shape[-3:], mode="trilinear", align_corners=False)
        value = self.dec2(torch.cat((value, e2), 1))
        value = F.interpolate(value, e1.shape[-3:], mode="trilinear", align_corners=False)
        value = self.dec1(torch.cat((value, e1), 1))
        value = F.interpolate(value, e0.shape[-3:], mode="trilinear", align_corners=False)
        raw = self.head(self.dec0(torch.cat((value, e0), 1)))
        residual = self.maximum_residual * torch.tanh(raw[:, :3])
        alpha = raw[:, 3:4].sigmoid()
        parent = visual[:, :3]
        return (parent + alpha * residual).clamp(0, 1), alpha


def parameter_count(base_channels: int = 32) -> int:
    return sum(value.numel() for value in VideoDetailRefinerV180(base_channels).parameters())
