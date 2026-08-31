"""Two frozen-parent specialists for glyph transport and black gripper structure.

The glyph expert predicts geometry, not characters: sharp pixels always come
from the last real observation.  The gripper expert predicts a binary structure
layer and a bounded RGB residual restricted to the predicted/source/parent
structure neighbourhood.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(input_channels + hidden_channels, 2 * hidden_channels, 3, padding=1)
        self.candidate = nn.Conv2d(input_channels + hidden_channels, hidden_channels, 3, padding=1)

    def forward(self, value: torch.Tensor, hidden: torch.Tensor | None) -> torch.Tensor:
        if hidden is None:
            hidden = value.new_zeros(value.shape[0], self.hidden_channels, *value.shape[-2:])
        reset, update = self.gates(torch.cat((value, hidden), 1)).sigmoid().chunk(2, 1)
        candidate = torch.tanh(self.candidate(torch.cat((value, reset * hidden), 1)))
        return (1.0 - update) * hidden + update * candidate


class _SequenceEncoder(nn.Module):
    def __init__(self, input_channels: int, base_channels: int) -> None:
        super().__init__()
        c = base_channels
        groups = max(c // 4, 1)
        self.visual = nn.Sequential(
            nn.Conv2d(input_channels, c, 5, padding=2), nn.GroupNorm(groups, c), nn.SiLU(),
            nn.Conv2d(c, 2 * c, 4, stride=2, padding=1), nn.GroupNorm(2 * groups, 2 * c), nn.SiLU(),
            nn.Conv2d(2 * c, 2 * c, 4, stride=2, padding=1), nn.GroupNorm(2 * groups, 2 * c), nn.SiLU(),
        )
        self.action = nn.Sequential(nn.Linear(7, c), nn.SiLU(), nn.Linear(c, c), nn.SiLU())
        self.arm = nn.Embedding(2, c)
        self.recurrent = ConvGRUCell(4 * c, 2 * c)

    def step(self, visual: torch.Tensor, action: torch.Tensor, arm_id: torch.Tensor,
             hidden: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.visual(visual)
        action_feature = self.action(action)[:, :, None, None].expand(-1, -1, *encoded.shape[-2:])
        arm_feature = self.arm(arm_id)[:, :, None, None].expand(-1, -1, *encoded.shape[-2:])
        hidden = self.recurrent(torch.cat((encoded, action_feature, arm_feature), 1), hidden)
        return torch.cat((encoded, hidden), 1), hidden


class TinyGlyphMotionExpert(nn.Module):
    """Predict target-to-source affine transforms and visibility confidence."""

    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        c = base_channels
        # source/current/delta RGB (9), source glyph, source beam, parent beam.
        self.encoder = _SequenceEncoder(12, c)
        self.head = nn.Sequential(
            nn.Conv2d(4 * c, 2 * c, 3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(2 * c, c), nn.SiLU(),
            nn.Linear(c, 7),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, source: torch.Tensor, parent: torch.Tensor,
                actions: torch.Tensor, arm_id: torch.Tensor,
                source_glyph: torch.Tensor, source_beam: torch.Tensor,
                parent_beam: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = None
        previous = source
        matrices, confidences = [], []
        identity = source.new_tensor(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))[None]
        for time in range(parent.shape[1]):
            current = parent[:, time]
            visual = torch.cat((source, current, current - previous, source_glyph,
                                source_beam, parent_beam[:, time]), 1)
            feature, hidden = self.encoder.step(visual, actions[:, time], arm_id, hidden)
            raw = self.head(feature)
            delta = torch.tanh(raw[:, :6]).view(-1, 2, 3)
            scale = source.new_tensor(((0.18, 0.18, 0.42), (0.18, 0.18, 0.42)))[None]
            matrices.append(identity + scale * delta)
            confidences.append(raw[:, 6].sigmoid())
            previous = current
        return torch.stack(matrices, 1), torch.stack(confidences, 1)


class TinyBlackGripperExpert(nn.Module):
    """Binary black-structure predictor with a bounded photometric residual."""

    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        c = base_channels
        # source/current/delta RGB (9), source black, parent black, parent bottle.
        self.encoder = _SequenceEncoder(12, c)
        groups = max(c // 4, 1)
        self.decoder = nn.Sequential(
            nn.Conv2d(4 * c, 2 * c, 3, padding=1), nn.GroupNorm(2 * groups, 2 * c), nn.SiLU(),
            nn.Conv2d(2 * c, c, 3, padding=1), nn.GroupNorm(groups, c), nn.SiLU(),
            nn.Conv2d(c, 4, 1),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, source: torch.Tensor, parent: torch.Tensor,
                actions: torch.Tensor, arm_id: torch.Tensor,
                source_black: torch.Tensor, parent_black: torch.Tensor,
                parent_bottle: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = None
        previous = source
        logits, residuals = [], []
        for time in range(parent.shape[1]):
            current = parent[:, time]
            visual = torch.cat((source, current, current - previous, source_black,
                                parent_black[:, time], parent_bottle[:, time]), 1)
            feature, hidden = self.encoder.step(visual, actions[:, time], arm_id, hidden)
            decoded = F.interpolate(self.decoder(feature), current.shape[-2:], mode="bilinear",
                                    align_corners=False)
            prior = parent_black[:, time].clamp(0.02, 0.98)
            prior_logit = torch.logit(prior)
            logits.append(prior_logit + decoded[:, :1])
            residuals.append(0.35 * torch.tanh(decoded[:, 1:]))
            previous = current
        return torch.stack(logits, 1), torch.stack(residuals, 1)


def warp_observed_glyph(source: torch.Tensor, source_mask: torch.Tensor,
                        matrices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Warp one observed crop into every predicted frame."""
    batch, time = matrices.shape[:2]
    source_frames = source[:, None].expand(-1, time, -1, -1, -1).flatten(0, 1)
    mask_frames = source_mask[:, None].expand(-1, time, -1, -1, -1).flatten(0, 1)
    grid = F.affine_grid(matrices.flatten(0, 1), source_frames.shape, align_corners=False)
    warped_source = F.grid_sample(source_frames, grid, mode="bilinear", padding_mode="border",
                                  align_corners=False)
    warped_mask = F.grid_sample(mask_frames, grid, mode="bilinear", padding_mode="zeros",
                                align_corners=False)
    return warped_source.unflatten(0, (batch, time)), warped_mask.unflatten(0, (batch, time))


def render_glyph(parent_clean: torch.Tensor, source: torch.Tensor, source_mask: torch.Tensor,
                 parent_beam: torch.Tensor, matrices: torch.Tensor,
                 confidence: torch.Tensor, strength: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    warped_source, warped_mask = warp_observed_glyph(source, source_mask, matrices)
    alpha = (strength * confidence[:, :, None, None, None] * warped_mask * parent_beam).clamp(0, 1)
    return parent_clean * (1 - alpha) + warped_source * alpha, alpha


def render_black_gripper(parent: torch.Tensor, source_black: torch.Tensor,
                         parent_black: torch.Tensor, logits: torch.Tensor,
                         residual: torch.Tensor, strength: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probability = logits.sigmoid()
    union = torch.maximum(torch.maximum(probability, parent_black),
                          source_black[:, None].expand_as(probability))
    support = F.max_pool3d(union, kernel_size=(1, 7, 7), stride=1, padding=(0, 3, 3))
    output = (parent + strength * support * residual).clamp(0, 1)
    return output, probability, support


def parameter_counts() -> dict[str, int]:
    return {
        "glyph": sum(value.numel() for value in TinyGlyphMotionExpert().parameters()),
        "black_gripper": sum(value.numel() for value in TinyBlackGripperExpert().parameters()),
    }
