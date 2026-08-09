"""v10.1 candidate risk router with cross-source and flow-geometry evidence."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .candidate_risk_router_v10 import CandidateRiskRouterV10


class CandidateRiskRouterV101(CandidateRiskRouterV10):
    """Add source consensus, variance, flow roughness, and folding evidence."""

    feature_channels = 27

    @staticmethod
    def _flow_geometry(flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Flow is a backward sampling map offset. Spatial discontinuity and a
        # non-positive mapping Jacobian are observable indicators of unreliable warps.
        du_dx = F.pad(flow[:, :, :, 0, :, 1:] - flow[:, :, :, 0, :, :-1], (0, 1, 0, 0))
        du_dy = F.pad(flow[:, :, :, 0, 1:, :] - flow[:, :, :, 0, :-1, :], (0, 0, 0, 1))
        dv_dx = F.pad(flow[:, :, :, 1, :, 1:] - flow[:, :, :, 1, :, :-1], (0, 1, 0, 0))
        dv_dy = F.pad(flow[:, :, :, 1, 1:, :] - flow[:, :, :, 1, :-1, :], (0, 0, 0, 1))
        roughness = (du_dx.abs() + du_dy.abs() + dv_dx.abs() + dv_dy.abs())[:, :, :, None] / 32.0
        determinant = (1 + du_dx) * (1 + dv_dy) - du_dy * dv_dx
        folding = F.relu(0.10 - determinant)[:, :, :, None].clamp_max(8.0) / 8.0
        return roughness, folding

    def observable_features(
        self,
        context: torch.Tensor,
        parent: torch.Tensor,
        transported: torch.Tensor,
        refined_flow: torch.Tensor,
        visibility_logits: torch.Tensor,
    ) -> torch.Tensor:
        base = self._base_observable_features(
            context, parent, transported, refined_flow, visibility_logits
        )
        median = transported.median(dim=2, keepdim=True).values
        consensus_distance = self._pool((transported - median).abs().mean(dim=3, keepdim=True))
        source_spread = transported.std(dim=2, keepdim=True, unbiased=False).mean(dim=3, keepdim=True)
        source_spread = self._pool(source_spread).expand(-1, -1, transported.shape[2], -1, -1, -1)
        roughness, folding = self._flow_geometry(refined_flow)
        roughness = self._pool(roughness)
        folding = self._pool(folding)
        features = torch.cat((base, consensus_distance, source_spread, roughness, folding), dim=3)
        if features.shape[3] != self.feature_channels:
            raise RuntimeError(f"unexpected v10.1 risk feature shape: {features.shape}")
        return features

    def load_v10_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load v10 exactly and zero-initialize the four new evidence channels."""
        upgraded = self.state_dict()
        for key, value in state_dict.items():
            if key == "visual.0.weight":
                if value.shape[1] != 23 or upgraded[key].shape[1] != self.feature_channels:
                    raise ValueError(f"unexpected first-layer shape: {value.shape} -> {upgraded[key].shape}")
                upgraded[key].zero_()
                upgraded[key][:, : value.shape[1]].copy_(value)
            else:
                if key not in upgraded or upgraded[key].shape != value.shape:
                    raise ValueError(f"incompatible v10 parameter {key}: {value.shape}")
                upgraded[key].copy_(value)
        self.load_state_dict(upgraded, strict=True)
