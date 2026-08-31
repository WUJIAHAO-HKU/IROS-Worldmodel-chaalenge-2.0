"""Zero-initialized local high-frequency repair driven by explicit structure masks."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn

from .residual_unet import ConvBlock, DownBlock, UpBlock


class StructureGatedHighFrequencyRefiner(nn.Module):
    """Repair local texture without changing the frozen baseline at initialization."""

    prediction_frames = 8
    action_dim = 14

    def __init__(self, base_channels: int = 16, residual_scale: float = 0.03) -> None:
        super().__init__()
        if base_channels % 8:
            raise ValueError("base channels must be divisible by eight")
        self.base_channels = int(base_channels)
        self.residual_scale = float(residual_scale)
        # Baseline RGB, structure-parent RGB, last observation, disagreement,
        # last-observation highpass, structure mask, two motion magnitudes, horizon.
        input_channels = 3 + 3 + 3 + 3 + 3 + 1 + 1 + 1 + 1
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

    @staticmethod
    def highpass(value: torch.Tensor) -> torch.Tensor:
        flat = value.flatten(0, 1)
        blurred = functional.avg_pool2d(flat, 5, stride=1, padding=2, count_include_pad=False)
        return (flat - blurred).unflatten(0, value.shape[:2])

    def forward(
        self,
        context_last: torch.Tensor,
        baseline: torch.Tensor,
        structure_parent: torch.Tensor,
        structure_mask: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, steps, channels, height, width = baseline.shape
        if (steps, channels) != (self.prediction_frames, 3) or structure_parent.shape != baseline.shape:
            raise ValueError("parent predictions must be [B,8,3,H,W]")
        if context_last.shape != (batch, 1, 3, height, width):
            raise ValueError("context_last must be [B,1,3,H,W]")
        if structure_mask.shape != (batch, steps, 1, height, width):
            raise ValueError("structure_mask must be [B,8,1,H,W]")
        if future_actions.shape != (batch, steps, self.action_dim):
            raise ValueError("future_actions must be [B,8,14]")

        context = context_last.expand(-1, steps, -1, -1, -1)
        previous_baseline = torch.cat((context_last, baseline[:, :-1]), dim=1)
        previous_structure = torch.cat((context_last, structure_parent[:, :-1]), dim=1)
        baseline_motion = (baseline - previous_baseline).abs().mean(dim=2, keepdim=True)
        structure_motion = (structure_parent - previous_structure).abs().mean(dim=2, keepdim=True)
        disagreement = (baseline - structure_parent).abs()
        horizon = torch.linspace(0.0, 1.0, steps, device=baseline.device, dtype=baseline.dtype)
        horizon = horizon.view(1, steps, 1, 1, 1).expand(batch, -1, -1, height, width)
        visual = torch.cat(
            [baseline, structure_parent, context, disagreement, self.highpass(context), structure_mask, baseline_motion, structure_motion, horizon],
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
        learned_gate = torch.sigmoid(output[:, :, :1])
        raw_residual = self.residual_scale * torch.tanh(output[:, :, 1:])
        high_frequency_residual = self.highpass(raw_residual)
        disagreement_support = (disagreement.mean(dim=2, keepdim=True) / 0.015).clamp(0.0, 1.0)
        motion_support = ((baseline_motion + structure_motion) / 0.03).clamp(0.0, 1.0)
        support = structure_mask.clamp(0.0, 1.0) * (0.25 + 0.75 * disagreement_support) * (0.25 + 0.75 * motion_support)
        correction = support * learned_gate * high_frequency_residual
        prediction = baseline + correction
        return prediction, learned_gate, correction, support
