"""Four-class recurrent head for bottle, black pad and grey gripper support."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .contact_occlusion_head_v13 import ConvGRUCell


class ContactStructureHeadV135(nn.Module):
    def __init__(self, base_channels: int = 32) -> None:
        super().__init__(); c = base_channels; self.base_channels = c
        # RGB(last/current/delta)=9 plus source/parent bottle-black-grey priors=6.
        self.image_encoder = nn.Sequential(
            nn.Conv2d(15, c, 5, padding=2), nn.GroupNorm(8, c), nn.SiLU(),
            nn.Conv2d(c, 2 * c, 4, stride=2, padding=1), nn.GroupNorm(8, 2 * c), nn.SiLU(),
            nn.Conv2d(2 * c, 2 * c, 4, stride=2, padding=1), nn.GroupNorm(8, 2 * c), nn.SiLU(),
        )
        self.action_encoder = nn.Sequential(nn.Linear(7, c), nn.SiLU(), nn.Linear(c, c), nn.SiLU())
        self.arm_embedding = nn.Embedding(2, c)
        self.recurrent = ConvGRUCell(4 * c, 2 * c)
        self.shared_decoder = nn.Sequential(
            nn.Conv2d(4 * c, 2 * c, 3, padding=1), nn.GroupNorm(8, 2 * c), nn.SiLU(),
            nn.Conv2d(2 * c, c, 3, padding=1), nn.GroupNorm(8, c), nn.SiLU(),
            nn.Conv2d(c, 4, 1),
        )
        self.arm_residual = nn.ModuleList((
            nn.Sequential(nn.Conv2d(4 * c, c, 3, padding=1), nn.SiLU(), nn.Conv2d(c, 4, 1)),
            nn.Sequential(nn.Conv2d(4 * c, c, 3, padding=1), nn.SiLU(), nn.Conv2d(c, 4, 1)),
        ))
        for expert in self.arm_residual:
            nn.init.zeros_(expert[-1].weight); nn.init.zeros_(expert[-1].bias)

    def forward(self, last_observation: torch.Tensor, parent_prediction: torch.Tensor,
                active_actions: torch.Tensor, arm_id: torch.Tensor,
                source_semantic: torch.Tensor, parent_semantic: torch.Tensor) -> torch.Tensor:
        hidden = None; previous = last_observation; outputs = []
        arm_feature = self.arm_embedding(arm_id)[:, :, None, None]
        for time in range(parent_prediction.shape[1]):
            current = parent_prediction[:, time]
            visual = torch.cat((last_observation, current, current - previous,
                                source_semantic, parent_semantic[:, time]), dim=1)
            encoded = self.image_encoder(visual)
            action = self.action_encoder(active_actions[:, time])[:, :, None, None]
            action = action.expand(-1, -1, *encoded.shape[-2:])
            arm = arm_feature.expand(-1, -1, *encoded.shape[-2:])
            hidden = self.recurrent(torch.cat((encoded, action, arm), dim=1), hidden)
            feature = torch.cat((encoded, hidden), dim=1)
            logits = self.shared_decoder(feature)
            experts = torch.stack([head(feature) for head in self.arm_residual], dim=1)
            logits = logits + experts[torch.arange(len(arm_id), device=arm_id.device), arm_id]
            outputs.append(F.interpolate(logits, current.shape[-2:], mode="bilinear",
                                         align_corners=False))
            previous = current
        return torch.stack(outputs, dim=1)
