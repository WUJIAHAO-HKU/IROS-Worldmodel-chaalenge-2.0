"""Blockwise router over a protected parent and predicted texture transports."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, source: int, target: int) -> None:
        super().__init__()
        groups = min(8, target)
        self.net = nn.Sequential(
            nn.Conv2d(source, target, 3, padding=1), nn.GroupNorm(groups, target), nn.SiLU(),
            nn.Conv2d(target, target, 3, padding=1), nn.GroupNorm(groups, target), nn.SiLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class HybridTransportRouter(nn.Module):
    """Select 4x4 RGB blocks without synthesizing or blending their texture."""

    candidates = 7
    prediction_frames = 8
    active_action_dim = 7
    routing_resolution = 64
    feature_channels = 42

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        self.base_channels = int(base_channels)
        c = self.base_channels
        self.enc0 = ConvBlock(self.feature_channels, c)
        self.enc1 = ConvBlock(c, 2 * c)
        self.enc2 = ConvBlock(2 * c, 4 * c)
        self.middle = ConvBlock(4 * c, 4 * c)
        self.action_gru = nn.GRU(self.active_action_dim, 4 * c, batch_first=True)
        self.arm_embedding = nn.Embedding(2, 4 * c)
        self.horizon_embedding = nn.Linear(1, 4 * c)
        self.film = nn.Linear(4 * c, 8 * c)
        self.dec1 = ConvBlock(6 * c, 2 * c)
        self.dec0 = ConvBlock(3 * c, c)
        self.logits = nn.Conv2d(c, self.candidates, 1)
        nn.init.zeros_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)
        with torch.no_grad():
            # Parent still wins exactly at initialization, but the router can
            # overturn the prior within a short pilot instead of saturating.
            self.logits.bias[0] = 0.25

    @staticmethod
    def active_arm_actions(actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Factor [B,12,14] into the one changing 7-D arm plus arm identity."""
        if actions.ndim != 3 or actions.shape[1:] != (12, 14):
            raise ValueError("actions must be [B,12,14]")
        delta = actions[:, 1:] - actions[:, :-1]
        activity = torch.stack((delta[:, :, :7].abs().mean((1, 2)), delta[:, :, 7:].abs().mean((1, 2))), dim=1)
        arm = activity.argmax(dim=1)
        arms = actions.reshape(actions.shape[0], 12, 2, 7)
        active = arms.gather(2, arm[:, None, None, None].expand(-1, 12, 1, 7)).squeeze(2)
        return active[:, 4:], arm

    @staticmethod
    def _resize(value: torch.Tensor, size: int) -> torch.Tensor:
        leading = value.shape[:-3]
        result = F.interpolate(value.flatten(0, len(leading) - 1), (size, size), mode="bilinear", align_corners=False)
        return result.unflatten(0, leading)

    def visual_features(self, candidates: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, t, k, channels, height, width = candidates.shape
        if (t, k, channels) != (8, self.candidates, 3) or context.shape[:3] != (b, 5, 3):
            raise ValueError("expected candidates [B,8,7,3,H,W] and context [B,5,3,H,W]")
        size = self.routing_resolution
        small = self._resize(candidates, size)
        observed = self._resize(context, size)
        parent = small[:, :, :1]
        disagreement = (small[:, :, 1:] - parent).abs().mean(dim=3)
        last = observed[:, -1]
        context_motion = torch.stack(
            ((observed[:, -1] - observed[:, -2]).abs().mean(dim=1),
             (observed[:, -1] - observed[:, 0]).abs().mean(dim=1)), dim=1,
        )
        previous = torch.cat((last[:, None, None].expand(-1, 1, k, -1, -1, -1), small[:, :-1]), dim=1)
        candidate_motion = (small - previous).abs().mean(dim=3)
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, size, device=small.device, dtype=small.dtype),
            torch.linspace(-1, 1, size, device=small.device, dtype=small.dtype), indexing="ij",
        )
        coordinates = torch.stack((x, y))[None, None].expand(b, t, -1, -1, -1)
        horizon = torch.linspace(1 / t, 1, t, device=small.device, dtype=small.dtype)[None, :, None, None, None]
        horizon = horizon.expand(b, -1, -1, size, size)
        last = last[:, None].expand(-1, t, -1, -1, -1)
        context_motion = context_motion[:, None].expand(-1, t, -1, -1, -1)
        features = torch.cat(
            (small.flatten(2, 3), disagreement, last, context_motion, candidate_motion, coordinates, horizon), dim=2
        )
        if features.shape[2] != self.feature_channels:
            raise RuntimeError(f"internal feature count mismatch: {features.shape}")
        return features.flatten(0, 1)

    def forward(
        self,
        candidates: torch.Tensor,
        context: torch.Tensor,
        active_actions: torch.Tensor,
        arm_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, _, _, height, width = candidates.shape
        if active_actions.shape != (b, t, 7) or arm_id.shape != (b,):
            raise ValueError("active_actions must be [B,8,7] and arm_id [B]")
        visual = self.visual_features(candidates, context)
        e0 = self.enc0(visual)
        e1 = self.enc1(F.avg_pool2d(e0, 2))
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        middle = self.middle(e2)
        action_state = self.action_gru(active_actions)[0]
        horizon = torch.linspace(1 / t, 1, t, device=active_actions.device, dtype=active_actions.dtype)
        condition = action_state + self.arm_embedding(arm_id)[:, None] + self.horizon_embedding(horizon[None, :, None]).expand(b, -1, -1)
        scale, shift = self.film(condition.flatten(0, 1)).chunk(2, dim=1)
        middle = middle * (1 + scale[..., None, None]) + shift[..., None, None]
        decoded = F.interpolate(middle, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        decoded = self.dec1(torch.cat((decoded, e1), dim=1))
        decoded = F.interpolate(decoded, size=e0.shape[-2:], mode="bilinear", align_corners=False)
        logits = self.logits(self.dec0(torch.cat((decoded, e0), dim=1))).unflatten(0, (b, t))
        weights = logits.softmax(dim=2)
        choice = weights.argmax(dim=2)
        block = height // self.routing_resolution
        full_choice = choice.repeat_interleave(block, 2).repeat_interleave(block, 3)
        hard = F.one_hot(full_choice, num_classes=self.candidates).permute(0, 1, 4, 2, 3).to(candidates.dtype)
        full_weights = weights.repeat_interleave(block, 3).repeat_interleave(block, 4)
        selection = hard + full_weights - full_weights.detach() if self.training else hard
        prediction = (selection[:, :, :, None] * candidates).sum(dim=2)
        return prediction, logits, weights
