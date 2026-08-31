"""Explicit texture reprojection and structure completion for AR predictions."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn

from .residual_unet import ConvBlock, DownBlock, UpBlock


class AutoregressiveTextureReprojection(nn.Module):
    """Reproject five observed views onto an AR rollout, then fill disocclusions."""

    context_frames = 5
    history_actions = 4
    prediction_frames = 8
    action_dim = 14
    max_flow_pixels = 128.0

    def __init__(self, base_channels: int = 32, residual_scale: float = 0.10) -> None:
        super().__init__()
        if base_channels < 8 or base_channels % 8:
            raise ValueError("base_channels must be a positive multiple of eight")
        self.base_channels = int(base_channels)
        self.residual_scale = float(residual_scale)
        base = self.base_channels
        # Five observed RGB frames, AR RGB, AR displacement from the last
        # observation, displacement magnitude, coordinates, and horizon.
        input_channels = 15 + 3 + 3 + 1 + 2 + 1
        self.enc0 = ConvBlock(input_channels, base)
        self.enc1 = DownBlock(base, base * 2)
        self.enc2 = DownBlock(base * 2, base * 3)
        self.enc3 = DownBlock(base * 3, base * 4)
        self.enc4 = DownBlock(base * 4, base * 5)
        self.middle = ConvBlock(base * 5, base * 5)
        self.action_gru = nn.GRU(self.action_dim, base * 5, batch_first=True)
        self.action_norm = nn.LayerNorm(base * 5)
        self.film = nn.Linear(base * 5, base * 10)
        self.up3 = UpBlock(base * 5, base * 4, base * 4)
        self.up2 = UpBlock(base * 4, base * 3, base * 3)
        self.up1 = UpBlock(base * 3, base * 2, base * 2)
        self.up0 = UpBlock(base * 2, base, base)
        # Five backward flows, five source logits, a visibility gate, and RGB
        # structure completion. Gate initialization makes step zero reproduce
        # the audited AR parent to within quantization noise.
        channels = self.context_frames * 2 + self.context_frames + 1 + 3
        self.output = nn.Conv2d(base, channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        with torch.no_grad():
            # Prefer the newest view without saturating the source softmax;
            # expose a small but trainable reprojection path from step zero.
            self.output.bias[self.context_frames * 2 + self.context_frames - 1] = 4.0
            self.output.bias[self.context_frames * 3] = -5.0

    @staticmethod
    def coordinates(batch: int, height: int, width: int, device, dtype) -> torch.Tensor:
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack((x, y)).unsqueeze(0).expand(batch, -1, -1, -1)

    @staticmethod
    def warp(images: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Warp [B,S,3,H,W] observations with target-to-source pixel flows."""
        batch, steps, sources, _, height, width = flow.shape
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=images.device, dtype=images.dtype),
            torch.linspace(-1.0, 1.0, width, device=images.device, dtype=images.dtype),
            indexing="ij",
        )
        base = torch.stack((x, y), dim=-1).view(1, 1, 1, height, width, 2)
        scale = torch.tensor((2.0 / (width - 1), 2.0 / (height - 1)), device=images.device, dtype=images.dtype)
        grid = base + flow.permute(0, 1, 2, 4, 5, 3) * scale
        expanded = images[:, None].expand(-1, steps, -1, -1, -1, -1)
        expanded = expanded.reshape(batch * steps * sources, 3, height, width)
        warped = functional.grid_sample(
            expanded,
            grid.reshape(batch * steps * sources, height, width, 2),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return warped.reshape(batch, steps, sources, 3, height, width)

    def forward(
        self,
        context_frames: torch.Tensor,
        autoregressive: torch.Tensor,
        actions: torch.Tensor,
        return_components: bool = False,
    ):
        batch, frames, channels, height, width = context_frames.shape
        if (frames, channels) != (self.context_frames, 3):
            raise ValueError("context_frames must be [B,5,3,H,W]")
        if autoregressive.shape != (batch, self.prediction_frames, 3, height, width):
            raise ValueError("autoregressive must be [B,8,3,H,W]")
        if actions.shape != (batch, self.history_actions + self.prediction_frames, self.action_dim):
            raise ValueError("actions must be [B,12,14]")
        steps = self.prediction_frames
        context = context_frames.flatten(1, 2)[:, None].expand(-1, steps, -1, -1, -1)
        anchor = context_frames[:, -1:,].expand(-1, steps, -1, -1, -1)
        displacement = autoregressive - anchor
        magnitude = displacement.abs().mean(dim=2, keepdim=True)
        coordinates = self.coordinates(batch, height, width, autoregressive.device, autoregressive.dtype)
        coordinates = coordinates[:, None].expand(-1, steps, -1, -1, -1)
        horizon = torch.linspace(0.0, 1.0, steps, device=autoregressive.device, dtype=autoregressive.dtype)
        horizon = horizon.view(1, steps, 1, 1, 1).expand(batch, -1, -1, height, width)
        visual = torch.cat((context, autoregressive, displacement, magnitude, coordinates, horizon), dim=2).flatten(0, 1)
        e0 = self.enc0(visual)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        states, _ = self.action_gru(actions)
        condition = self.action_norm(states[:, self.history_actions :]).flatten(0, 1)
        scale, shift = self.film(condition).chunk(2, dim=1)
        middle = self.middle(e4) * (1.0 + scale[..., None, None]) + shift[..., None, None]
        decoded = self.up3(middle, e3)
        decoded = self.up2(decoded, e2)
        decoded = self.up1(decoded, e1)
        decoded = self.up0(decoded, e0)
        output = self.output(decoded).unflatten(0, (batch, steps))
        flow_end = self.context_frames * 2
        source_end = flow_end + self.context_frames
        flow = torch.tanh(output[:, :, :flow_end]).reshape(batch, steps, self.context_frames, 2, height, width)
        flow = flow * self.max_flow_pixels
        source_weight = torch.softmax(output[:, :, flow_end:source_end], dim=2).unsqueeze(3)
        gate = torch.sigmoid(output[:, :, source_end : source_end + 1])
        residual = self.residual_scale * torch.tanh(output[:, :, source_end + 1 :])
        reprojected = (source_weight * self.warp(context_frames, flow)).sum(dim=2)
        prediction = autoregressive + gate * (reprojected - autoregressive) + residual
        if return_components:
            return prediction, flow, source_weight, gate, residual, reprojected
        return prediction
