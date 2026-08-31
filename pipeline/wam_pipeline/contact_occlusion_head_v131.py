"""Balanced dual-arm recurrent contact segmentation head."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .contact_occlusion_head_v13 import CONTACT_REGION, ConvGRUCell


class ContactOcclusionHeadV131(nn.Module):
    """Shared temporal geometry with explicit arm conditioning and arm experts."""

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__(); c = base_channels; self.base_channels = c
        self.image_encoder = nn.Sequential(
            nn.Conv2d(9, c, 5, padding=2), nn.GroupNorm(8, c), nn.SiLU(),
            nn.Conv2d(c, 2 * c, 4, stride=2, padding=1), nn.GroupNorm(8, 2 * c), nn.SiLU(),
            nn.Conv2d(2 * c, 2 * c, 4, stride=2, padding=1), nn.GroupNorm(8, 2 * c), nn.SiLU(),
        )
        self.action_encoder = nn.Sequential(nn.Linear(7, c), nn.SiLU(), nn.Linear(c, c), nn.SiLU())
        self.arm_embedding = nn.Embedding(2, c)
        self.recurrent = ConvGRUCell(4 * c, 2 * c)
        self.shared_decoder = nn.Sequential(
            nn.Conv2d(4 * c, 2 * c, 3, padding=1), nn.GroupNorm(8, 2 * c), nn.SiLU(),
            nn.Conv2d(2 * c, c, 3, padding=1), nn.GroupNorm(8, c), nn.SiLU(),
            nn.Conv2d(c, 3, 1),
        )
        self.arm_residual = nn.ModuleList((
            nn.Sequential(nn.Conv2d(4 * c, c, 3, padding=1), nn.SiLU(), nn.Conv2d(c, 3, 1)),
            nn.Sequential(nn.Conv2d(4 * c, c, 3, padding=1), nn.SiLU(), nn.Conv2d(c, 3, 1)),
        ))
        for expert in self.arm_residual:
            nn.init.zeros_(expert[-1].weight); nn.init.zeros_(expert[-1].bias)

    def forward(self, last_observation: torch.Tensor, parent_prediction: torch.Tensor,
                active_actions: torch.Tensor, arm_id: torch.Tensor) -> torch.Tensor:
        if active_actions.shape[-1] != 7 or arm_id.ndim != 1:
            raise ValueError("expected active_actions [B,8,7] and arm_id [B]")
        hidden = None; previous = last_observation; outputs = []
        arm_feature = self.arm_embedding(arm_id)[:, :, None, None]
        for time in range(parent_prediction.shape[1]):
            current = parent_prediction[:, time]
            encoded = self.image_encoder(torch.cat((last_observation, current, current - previous), dim=1))
            action = self.action_encoder(active_actions[:, time])[:, :, None, None]
            action = action.expand(-1, -1, *encoded.shape[-2:])
            arm = arm_feature.expand(-1, -1, *encoded.shape[-2:])
            hidden = self.recurrent(torch.cat((encoded, action, arm), dim=1), hidden)
            feature = torch.cat((encoded, hidden), dim=1)
            shared = self.shared_decoder(feature)
            experts = torch.stack([head(feature) for head in self.arm_residual], dim=1)
            selected = experts[torch.arange(len(arm_id), device=arm_id.device), arm_id]
            logits = shared + selected
            outputs.append(F.interpolate(logits, current.shape[-2:], mode="bilinear", align_corners=False))
            previous = current
        return torch.stack(outputs, dim=1)

