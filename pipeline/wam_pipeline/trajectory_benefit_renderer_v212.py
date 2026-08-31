"""Trajectory-level discrete benefit renderer for coherent texture recovery."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


ALPHA_LEVELS = (0., .025, .05, .10, .20, .40)


class TrajectoryBenefitRenderer(nn.Module):
    patch_size = 8

    def __init__(self) -> None:
        super().__init__()
        # base, selected, disagreement, high-frequency disagreement (12),
        # five selector probabilities, action (14), horizon and active arm.
        self.network = nn.Sequential(
            nn.Conv3d(33, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv3d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv3d(64, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.GELU(),
            nn.Conv3d(48, len(ALPHA_LEVELS), 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        with torch.no_grad():
            self.network[-1].bias[0] = 2.0

    def forward(self, base: torch.Tensor, selected: torch.Tensor,
                selector_probabilities: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        batch, time, _, height, width = base.shape
        expected = (batch, time, 5, height // self.patch_size, width // self.patch_size)
        if selector_probabilities.shape != expected or actions.shape != (batch, time, 14):
            raise ValueError("trajectory renderer input shape")
        blur = lambda value: F.avg_pool2d(value, 5, 1, 2, count_include_pad=False)
        high_difference = (selected.flatten(0, 1) - blur(selected.flatten(0, 1))) \
                          - (base.flatten(0, 1) - blur(base.flatten(0, 1)))
        high_difference = high_difference.reshape_as(base)
        visual = torch.cat((base, selected, (selected - base).abs(), high_difference.abs()), 2)
        visual = F.avg_pool2d(visual.flatten(0, 1), self.patch_size, self.patch_size)
        visual = visual.reshape(batch, time, 12, height // self.patch_size, width // self.patch_size)
        horizon = torch.linspace(1 / time, 1, time, device=base.device, dtype=base.dtype)
        horizon = horizon[None, :, None].expand(batch, -1, -1)
        delta = torch.cat((actions[:, :1], actions[:, 1:] - actions[:, :-1]), 1).abs()
        arm = (delta[:, :, 7:].mean(2) > delta[:, :, :7].mean(2)).to(base.dtype)[:, :, None]
        condition = torch.cat((actions, horizon, arm), 2)
        condition = condition[:, :, :, None, None].expand(-1, -1, -1, *visual.shape[-2:])
        features = torch.cat((visual, selector_probabilities, condition), 2)
        return self.network(features.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)


def render(base: torch.Tensor, selected: torch.Tensor, logits: torch.Tensor,
           suppress_first: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    source = logits.argmax(2)
    levels = torch.tensor(ALPHA_LEVELS, device=base.device, dtype=base.dtype)
    alpha_patch = levels[source]
    if suppress_first:
        alpha_patch = alpha_patch.clone(); alpha_patch[:, :suppress_first] = 0
    alpha = F.interpolate(alpha_patch.flatten(0, 1)[:, None], size=base.shape[-2:], mode="nearest")
    alpha = alpha.reshape(*base.shape[:2], 1, *base.shape[-2:])
    blur = lambda value: F.avg_pool2d(value, 5, 1, 2, count_include_pad=False)
    delta = (selected.flatten(0, 1) - blur(selected.flatten(0, 1))) \
            - (base.flatten(0, 1) - blur(base.flatten(0, 1)))
    return (base + alpha * delta.reshape_as(base)).clamp(0, 1), alpha


def parameter_count() -> int:
    return sum(value.numel() for value in TrajectoryBenefitRenderer().parameters())
