"""Per-pixel, per-horizon fusion of autoregressive and direct-flow predictions."""

from __future__ import annotations

import torch
from torch import nn

from .residual_unet import ConvBlock, DownBlock, UpBlock


class LocalMotionTextureFusion(nn.Module):
    """Predict an alpha mask and bounded RGB correction from observable inputs."""

    prediction_frames = 8
    action_dim = 14

    def __init__(self, base_channels: int = 16, residual_scale: float = 0.05) -> None:
        super().__init__()
        if base_channels % 8:
            raise ValueError("base channels must be divisible by eight")
        self.base_channels = int(base_channels)
        self.residual_scale = float(residual_scale)
        # AR RGB, flow RGB, last observation, disagreement, two temporal
        # magnitudes, and normalized horizon position.
        input_channels = 3 + 3 + 3 + 3 + 1 + 1 + 1
        base = self.base_channels
        self.enc0 = ConvBlock(input_channels, base)
        self.enc1 = DownBlock(base, base * 2)
        self.enc2 = DownBlock(base * 2, base * 3)
        self.enc3 = DownBlock(base * 3, base * 4)
        self.middle = ConvBlock(base * 4, base * 4)
        self.action_embedding = nn.Sequential(nn.Linear(self.action_dim, base * 4), nn.SiLU())
        self.action_film = nn.Linear(base * 4, base * 8)
        self.up2 = UpBlock(base * 4, base * 3, base * 3)
        self.up1 = UpBlock(base * 3, base * 2, base * 2)
        self.up0 = UpBlock(base * 2, base, base)
        self.output = nn.Conv2d(base, 4, kernel_size=3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        context_last: torch.Tensor,
        autoregressive: torch.Tensor,
        direct_flow: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, steps, channels, height, width = autoregressive.shape
        if (steps, channels) != (self.prediction_frames, 3) or direct_flow.shape != autoregressive.shape:
            raise ValueError("parent predictions must be [B,8,3,H,W]")
        if context_last.shape != (batch, 1, 3, height, width):
            raise ValueError("context_last must be [B,1,3,H,W]")
        if future_actions.shape != (batch, steps, self.action_dim):
            raise ValueError("future_actions must be [B,8,14]")

        context = context_last.expand(-1, steps, -1, -1, -1)
        previous_ar = torch.cat([context_last, autoregressive[:, :-1]], dim=1)
        previous_flow = torch.cat([context_last, direct_flow[:, :-1]], dim=1)
        ar_motion = (autoregressive - previous_ar).abs().mean(dim=2, keepdim=True)
        flow_motion = (direct_flow - previous_flow).abs().mean(dim=2, keepdim=True)
        horizon = torch.linspace(0.0, 1.0, steps, device=autoregressive.device, dtype=autoregressive.dtype)
        horizon = horizon.view(1, steps, 1, 1, 1).expand(batch, -1, -1, height, width)
        visual = torch.cat(
            [autoregressive, direct_flow, context, (autoregressive - direct_flow).abs(), ar_motion, flow_motion, horizon],
            dim=2,
        ).flatten(0, 1)
        e0 = self.enc0(visual)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        middle = self.middle(e3)
        condition = self.action_embedding(future_actions.flatten(0, 1))
        scale, shift = self.action_film(condition).chunk(2, dim=1)
        middle = middle * (1.0 + scale[..., None, None]) + shift[..., None, None]
        decoded = self.up2(middle, e2)
        decoded = self.up1(decoded, e1)
        decoded = self.up0(decoded, e0)
        output = self.output(decoded).unflatten(0, (batch, steps))
        # Bias six starts at 99.75% AR, making the initial model an almost exact
        # reproduction of the safer parent while gradients remain available.
        alpha = torch.sigmoid(output[:, :, :1] + 6.0)
        residual = self.residual_scale * torch.tanh(output[:, :, 1:])
        support = ((autoregressive - direct_flow).abs().mean(dim=2, keepdim=True) / 0.08).clamp(0.0, 1.0)
        prediction = alpha * autoregressive + (1.0 - alpha) * direct_flow + support * residual
        return prediction, alpha, residual
