"""Candidate-conditioned uncertainty router for protected texture transport."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CandidateRiskRouterV10(nn.Module):
    """Predict per-candidate block advantage and uncertainty from observable evidence."""

    candidates = 5
    prediction_frames = 8
    block = 4
    feature_channels = 23
    advantage_scale = 5.0

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        c = int(base_channels)
        self.base_channels = c
        self.visual = nn.Sequential(
            nn.Conv2d(self.feature_channels, c, 3, padding=1),
            nn.GroupNorm(8, c), nn.SiLU(),
            nn.Conv2d(c, 3 * c // 2, 3, padding=1),
            nn.GroupNorm(8, 3 * c // 2), nn.SiLU(),
            nn.Conv2d(3 * c // 2, 2 * c, 3, padding=1),
            nn.GroupNorm(8, 2 * c), nn.SiLU(),
        )
        self.action_gru = nn.GRU(7, 2 * c, batch_first=True)
        self.arm_embedding = nn.Embedding(2, 2 * c)
        self.candidate_embedding = nn.Embedding(self.candidates, 2 * c)
        self.horizon_embedding = nn.Linear(1, 2 * c)
        self.film = nn.Linear(2 * c, 4 * c)
        self.output = nn.Sequential(
            nn.Conv2d(2 * c, c, 3, padding=1), nn.SiLU(), nn.Conv2d(c, 2, 1)
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        with torch.no_grad():
            self.output[-1].bias[0] = -0.25

    @staticmethod
    def _pool(value: torch.Tensor) -> torch.Tensor:
        shape = value.shape
        flat = value.flatten(0, -4)
        pooled = F.avg_pool2d(flat, 4, stride=4)
        return pooled.reshape(*shape[:-2], *pooled.shape[-2:])

    @staticmethod
    def _highpass_energy(value: torch.Tensor) -> torch.Tensor:
        shape = value.shape
        flat = value.flatten(0, -4)
        smooth = F.avg_pool2d(flat, 5, stride=1, padding=2, count_include_pad=False)
        energy = (flat - smooth).abs().mean(dim=1, keepdim=True)
        return energy.reshape(*shape[:-3], 1, *shape[-2:])

    @staticmethod
    def _coordinates(batch: int, steps: int, sources: int, device, dtype) -> torch.Tensor:
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, 64, device=device, dtype=dtype),
            torch.linspace(-1, 1, 64, device=device, dtype=dtype), indexing="ij",
        )
        return torch.stack((x, y))[None, None, None].expand(batch, steps, sources, -1, -1, -1)

    def _base_observable_features(
        self,
        context: torch.Tensor,
        parent: torch.Tensor,
        transported: torch.Tensor,
        refined_flow: torch.Tensor,
        visibility_logits: torch.Tensor,
    ) -> torch.Tensor:
        b, t, k, _, h, w = transported.shape
        if (t, k, h, w) != (8, 5, 256, 256):
            raise ValueError("transported must be [B,8,5,3,256,256]")
        parent_sources = parent[:, :, None].expand_as(transported)
        parent64 = self._pool(parent_sources)
        transported64 = self._pool(transported)
        delta64 = transported64 - parent64
        absolute_delta = delta64.abs().mean(dim=3, keepdim=True)
        parent_high = self._pool(self._highpass_energy(parent_sources))
        transported_high = self._pool(self._highpass_energy(transported))
        delta_high = self._pool(self._highpass_energy(transported - parent_sources))
        flow64 = self._pool(refined_flow) / 32.0
        magnitude = flow64.square().sum(dim=3, keepdim=True).add(1e-6).sqrt()
        visibility = self._pool(visibility_logits[:, :, :, None]).sigmoid()

        y, x = torch.meshgrid(
            torch.arange(256, device=refined_flow.device, dtype=refined_flow.dtype),
            torch.arange(256, device=refined_flow.device, dtype=refined_flow.dtype), indexing="ij",
        )
        mapped = torch.stack((x, y))[None, None, None] + refined_flow
        inside = ((mapped[:, :, :, 0] >= 0) & (mapped[:, :, :, 0] <= 255)
                  & (mapped[:, :, :, 1] >= 0) & (mapped[:, :, :, 1] <= 255))
        inside = self._pool(inside[:, :, :, None].to(parent.dtype))

        recent = (context[:, -1] - context[:, -2]).abs().mean(dim=1, keepdim=True)
        long = (context[:, -1] - context[:, 0]).abs().mean(dim=1, keepdim=True)
        context_motion = self._pool(torch.cat((recent, long), dim=1))
        context_motion = context_motion[:, None, None].expand(-1, t, k, -1, -1, -1)
        coordinates = self._coordinates(b, t, k, parent.device, parent.dtype)
        horizon = torch.linspace(1 / t, 1, t, device=parent.device, dtype=parent.dtype)
        horizon = horizon[None, :, None, None, None, None].expand(b, -1, k, 1, 64, 64)
        features = torch.cat(
            (parent64, transported64, delta64, absolute_delta, parent_high, transported_high,
             delta_high, flow64, magnitude, visibility, inside, context_motion,
             coordinates, horizon), dim=3,
        )
        return features

    def observable_features(
        self,
        context: torch.Tensor,
        parent: torch.Tensor,
        transported: torch.Tensor,
        refined_flow: torch.Tensor,
        visibility_logits: torch.Tensor,
    ) -> torch.Tensor:
        features = self._base_observable_features(
            context, parent, transported, refined_flow, visibility_logits
        )
        if features.shape[3] != self.feature_channels:
            raise RuntimeError(f"unexpected risk feature shape: {features.shape}")
        return features

    def forward(
        self,
        context: torch.Tensor,
        parent: torch.Tensor,
        transported: torch.Tensor,
        refined_flow: torch.Tensor,
        visibility_logits: torch.Tensor,
        active_actions: torch.Tensor,
        arm_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        features = self.observable_features(context, parent, transported, refined_flow, visibility_logits)
        b, t, k, _, _, _ = features.shape
        visual = self.visual(features.flatten(0, 2))
        action = self.action_gru(active_actions)[0] + self.arm_embedding(arm_id)[:, None]
        horizon = torch.linspace(1 / t, 1, t, device=context.device, dtype=context.dtype)
        action = action + self.horizon_embedding(horizon[None, :, None]).expand(b, -1, -1)
        candidates = torch.arange(k, device=context.device)
        condition = action[:, :, None] + self.candidate_embedding(candidates)[None, None]
        scale, shift = self.film(condition.flatten(0, 2)).chunk(2, dim=1)
        conditioned = visual * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        raw = self.output(conditioned).unflatten(0, (b, t, k))
        mean = raw[:, :, :, 0]
        uncertainty = F.softplus(raw[:, :, :, 1]) + 0.05
        return {
            "advantage_mean": mean * self.advantage_scale,
            "advantage_uncertainty": uncertainty * self.advantage_scale,
            "raw_mean": mean,
            "raw_uncertainty": uncertainty,
        }

    @staticmethod
    def safe_choice(mean: torch.Tensor, uncertainty: torch.Tensor, risk_weight: float, margin: float):
        safe = mean - float(risk_weight) * uncertainty
        best_score, best_source = safe.max(dim=2)
        choice = torch.where(best_score > float(margin), best_source + 1, 0)
        return choice, best_score
