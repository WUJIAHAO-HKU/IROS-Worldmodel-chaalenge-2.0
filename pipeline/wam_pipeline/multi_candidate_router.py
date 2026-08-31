"""Observable three-candidate router that preserves source pixels exactly."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(8, channels)
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.net(value)


class MultiCandidateRouter(nn.Module):
    """Predict spatial mixture weights from candidates and observed context only.

    The output is a convex mixture of the raw candidates. This deliberately has
    no RGB synthesis branch: selected small text and black structure cannot be
    blurred by a learned decoder.
    """

    feature_channels = 63

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.base_channels = c
        self.stem = nn.Conv2d(self.feature_channels, c, 3, padding=1)
        self.enc0 = nn.Sequential(ResidualBlock(c), ResidualBlock(c))
        self.down1 = nn.Conv2d(c, 2 * c, 4, stride=2, padding=1)
        self.enc1 = nn.Sequential(ResidualBlock(2 * c), ResidualBlock(2 * c))
        self.down2 = nn.Conv2d(2 * c, 4 * c, 4, stride=2, padding=1)
        self.mid = nn.Sequential(ResidualBlock(4 * c), ResidualBlock(4 * c), ResidualBlock(4 * c))
        self.up1 = nn.Conv2d(6 * c, 2 * c, 3, padding=1)
        self.dec1 = nn.Sequential(ResidualBlock(2 * c), ResidualBlock(2 * c))
        self.up0 = nn.Conv2d(3 * c, c, 3, padding=1)
        self.dec0 = nn.Sequential(ResidualBlock(c), ResidualBlock(c))
        self.logits = nn.Conv2d(c, 3, 1)
        nn.init.zeros_(self.logits.weight)
        with torch.no_grad():
            self.logits.bias.copy_(torch.tensor((8.0, 0.0, 0.0)))

    @staticmethod
    def features(candidates: torch.Tensor, context: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Return [B*T,63,H,W] observable routing features.

        candidates is [B,T,3,3,H,W], context is [B,5,3,H,W], and
        actions is the normalized future trajectory [B,T,14].
        """
        b, t, k, channels, height, width = candidates.shape
        if k != 3 or channels != 3:
            raise ValueError(f"expected three RGB candidates, got {tuple(candidates.shape)}")
        flat = candidates.reshape(b * t, 9, height, width)
        mean = candidates.mean(dim=2).reshape(b * t, 3, height, width)
        std = candidates.std(dim=2, unbiased=False).reshape(b * t, 3, height, width)
        pairwise = torch.cat(
            ((candidates[:, :, 0] - candidates[:, :, 1]).abs(),
             (candidates[:, :, 0] - candidates[:, :, 2]).abs(),
             (candidates[:, :, 1] - candidates[:, :, 2]).abs()), dim=2,
        ).reshape(b * t, 9, height, width)
        last_context = context[:, -1]
        previous = torch.cat((last_context[:, None, None].expand(-1, 1, 3, -1, -1, -1), candidates[:, :-1]), dim=1)
        delta = (candidates - previous).abs().reshape(b * t, 9, height, width)
        observed = context.flatten(1, 2)[:, None].expand(-1, t, -1, -1, -1).reshape(b * t, 15, height, width)
        action = actions[:, :, :, None, None].expand(-1, -1, -1, height, width).reshape(b * t, 14, height, width)
        horizon = torch.linspace(1.0 / t, 1.0, t, device=candidates.device, dtype=candidates.dtype)
        horizon = horizon[None, :, None, None, None].expand(b, -1, 1, height, width).reshape(b * t, 1, height, width)
        return torch.cat((flat, mean, std, pairwise, delta, observed, action, horizon), dim=1)

    def forward(self, candidates: torch.Tensor, context: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, _, _, height, width = candidates.shape
        x0 = self.enc0(self.stem(self.features(candidates, context, actions)))
        x1 = self.enc1(self.down1(x0))
        x2 = self.mid(self.down2(x1))
        x = F.interpolate(x2, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec1(self.up1(torch.cat((x, x1), dim=1)))
        x = F.interpolate(x, size=x0.shape[-2:], mode="bilinear", align_corners=False)
        logits = self.logits(self.dec0(self.up0(torch.cat((x, x0), dim=1))))
        weights = logits.softmax(dim=1).unflatten(0, (b, t))
        hard = F.one_hot(weights.argmax(dim=2), num_classes=3).permute(0, 1, 4, 2, 3).to(weights.dtype)
        # Straight-through hard routing: the forward path copies one candidate
        # exactly, while gradients still train the observable confidence map.
        selection = hard + weights - weights.detach() if self.training else hard
        prediction = (selection[:, :, :, None] * candidates).sum(dim=2)
        return prediction, weights
