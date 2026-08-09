"""Action-conditioned recurrent contact-layer segmentation for Track 2."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


CONTACT_REGION = (72, 232, 30, 210)


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        total = input_channels + hidden_channels
        self.gates = nn.Conv2d(total, 2 * hidden_channels, 3, padding=1)
        self.candidate = nn.Conv2d(total, hidden_channels, 3, padding=1)
        self.hidden_channels = hidden_channels

    def forward(self, value: torch.Tensor, hidden: torch.Tensor | None) -> torch.Tensor:
        if hidden is None:
            hidden = value.new_zeros(value.shape[0], self.hidden_channels, *value.shape[-2:])
        reset, update = self.gates(torch.cat((value, hidden), dim=1)).chunk(2, dim=1)
        reset, update = reset.sigmoid(), update.sigmoid()
        candidate = self.candidate(torch.cat((value, reset * hidden), dim=1)).tanh()
        return update * hidden + (1.0 - update) * candidate


class ContactOcclusionHeadV13(nn.Module):
    """Predict background/bottle/gripper labels jointly over all future frames."""

    def __init__(self, base_channels: int = 32, action_dimensions: int = 14) -> None:
        super().__init__()
        self.base_channels = base_channels
        self.action_dimensions = action_dimensions
        self.image_encoder = nn.Sequential(
            nn.Conv2d(9, base_channels, 5, padding=2), nn.GroupNorm(8, base_channels), nn.SiLU(),
            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 2), nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 2), nn.SiLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dimensions, base_channels), nn.SiLU(),
            nn.Linear(base_channels, base_channels), nn.SiLU(),
        )
        self.recurrent = ConvGRUCell(base_channels * 3, base_channels * 2)
        self.decoder = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.GroupNorm(8, base_channels * 2), nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels), nn.SiLU(),
            nn.Conv2d(base_channels, 3, 1),
        )

    def forward(
        self,
        last_observation: torch.Tensor,
        parent_prediction: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Inputs are B,C,H,W; B,T,C,H,W; B,T,A. Returns B,T,3,H,W."""
        hidden = None
        previous = last_observation
        outputs = []
        for time in range(parent_prediction.shape[1]):
            current = parent_prediction[:, time]
            image = torch.cat((last_observation, current, current - previous), dim=1)
            encoded = self.image_encoder(image)
            action = self.action_encoder(future_actions[:, time])[:, :, None, None]
            action = action.expand(-1, -1, *encoded.shape[-2:])
            hidden = self.recurrent(torch.cat((encoded, action), dim=1), hidden)
            low_logits = self.decoder(torch.cat((encoded, hidden), dim=1))
            outputs.append(F.interpolate(low_logits, size=current.shape[-2:], mode="bilinear", align_corners=False))
            previous = current
        return torch.stack(outputs, dim=1)

