"""v9.1 protected high-resolution flow, visibility, and confidence model."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .hybrid_transport_router import HybridTransportRouter
from .residual_unet import ConvBlock, DownBlock, UpBlock


class ProtectedLayeredFlowV91(nn.Module):
    """Refine frozen multi-source flow and hard-route texture against a v8 anchor."""

    prediction_frames = 8
    sources = 5
    candidates = 6
    routing_block = 4
    input_channels = 48

    def __init__(self, base_flow_model: nn.Module, base_channels: int = 16, max_residual_flow: float = 6.0) -> None:
        super().__init__()
        if base_channels % 8:
            raise ValueError("base_channels must be divisible by eight")
        self.base_flow_model = base_flow_model.requires_grad_(False)
        self.base_channels = int(base_channels)
        self.max_residual_flow = float(max_residual_flow)
        c = self.base_channels
        self.enc0 = ConvBlock(self.input_channels, c)
        self.enc1 = DownBlock(c, 2 * c)
        self.enc2 = DownBlock(2 * c, 3 * c)
        self.enc3 = DownBlock(3 * c, 4 * c)
        self.middle = ConvBlock(4 * c, 4 * c)
        self.action_gru = nn.GRU(7, 4 * c, batch_first=True)
        self.arm_embedding = nn.Embedding(2, 4 * c)
        self.horizon_embedding = nn.Linear(1, 4 * c)
        self.film = nn.Linear(4 * c, 8 * c)
        self.up2 = UpBlock(4 * c, 3 * c, 3 * c)
        self.up1 = UpBlock(3 * c, 2 * c, 2 * c)
        self.up0 = UpBlock(2 * c, c, c)
        # 5x2 residual flow, six candidate scores, five visibility logits.
        self.output = nn.Conv2d(c, 21, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        with torch.no_grad():
            self.output.bias[10] = 0.25  # protected v8 prior
            self.output.bias[16:] = 2.0  # sources initially considered visible

    @staticmethod
    def _coordinates(batch: int, steps: int, height: int, width: int, device, dtype) -> torch.Tensor:
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=device, dtype=dtype),
            torch.linspace(-1, 1, width, device=device, dtype=dtype), indexing="ij",
        )
        return torch.stack((x, y))[None, None].expand(batch, steps, -1, -1, -1)

    @staticmethod
    def _highpass(value: torch.Tensor) -> torch.Tensor:
        return value - F.avg_pool2d(value, 5, stride=1, padding=2, count_include_pad=False)

    @staticmethod
    def active_arm_actions(actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return HybridTransportRouter.active_arm_actions(actions)

    def _features(
        self,
        context: torch.Tensor,
        parent: torch.Tensor,
        initial_warps: torch.Tensor,
        base_flow: torch.Tensor,
    ) -> torch.Tensor:
        batch, steps, sources, _, height, width = initial_warps.shape
        observed = context.flatten(1, 2)[:, None].expand(-1, steps, -1, -1, -1)
        disagreement = (initial_warps - parent[:, :, None]).abs().mean(dim=3)
        flow_magnitude = base_flow.square().sum(dim=3).add(1e-6).sqrt() / max(height, width)
        context_motion = torch.stack(
            ((context[:, -1] - context[:, -2]).abs().mean(dim=1),
             (context[:, -1] - context[:, 0]).abs().mean(dim=1)), dim=1,
        )[:, None].expand(-1, steps, -1, -1, -1)
        horizon = torch.linspace(1 / steps, 1, steps, device=context.device, dtype=context.dtype)
        horizon = horizon[None, :, None, None, None].expand(batch, -1, -1, height, width)
        features = torch.cat(
            (parent, initial_warps.flatten(2, 3), observed, disagreement, flow_magnitude,
             context_motion, self._coordinates(batch, steps, height, width, context.device, context.dtype), horizon),
            dim=2,
        )
        if features.shape[2] != self.input_channels:
            raise RuntimeError(f"unexpected v9.1 feature shape: {features.shape}")
        return features.flatten(0, 1)

    def forward(
        self,
        context: torch.Tensor,
        base_actions: torch.Tensor,
        active_actions: torch.Tensor,
        arm_id: torch.Tensor,
        parent: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, steps, channels, height, width = parent.shape
        if (steps, channels, height, width) != (8, 3, 256, 256):
            raise ValueError("parent must be [B,8,3,256,256]")
        if context.shape != (batch, 5, 3, 256, 256) or base_actions.shape != (batch, 12, 14):
            raise ValueError("invalid context or base action shape")
        if active_actions.shape != (batch, 8, 7) or arm_id.shape != (batch,):
            raise ValueError("invalid factorized action shape")
        with torch.no_grad():
            _, base_flow, _ = self.base_flow_model(context, base_actions, return_flow=True)
            initial_warps = self.base_flow_model._warp(context, base_flow)
        visual = self._features(context, parent, initial_warps, base_flow)
        e0 = self.enc0(visual)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        middle = self.middle(e3)
        action_state = self.action_gru(active_actions)[0]
        horizon = torch.linspace(1 / steps, 1, steps, device=context.device, dtype=context.dtype)
        condition = action_state + self.arm_embedding(arm_id)[:, None]
        condition = condition + self.horizon_embedding(horizon[None, :, None]).expand(batch, -1, -1)
        scale, shift = self.film(condition.flatten(0, 1)).chunk(2, dim=1)
        middle = middle * (1 + scale[..., None, None]) + shift[..., None, None]
        decoded = self.up2(middle, e2)
        decoded = self.up1(decoded, e1)
        decoded = self.up0(decoded, e0)
        raw = self.output(decoded).unflatten(0, (batch, steps))
        residual_flow = torch.tanh(raw[:, :, :10]).reshape(batch, steps, 5, 2, height, width)
        residual_flow = residual_flow * self.max_residual_flow
        refined_flow = base_flow + residual_flow
        refined_warps = self.base_flow_model._warp(context, refined_flow)
        candidates = torch.cat((parent[:, :, None], refined_warps), dim=2)
        visibility_logits = raw[:, :, 16:]
        logits = raw[:, :, 10:16].clone()
        logits[:, :, 1:] = logits[:, :, 1:] + F.logsigmoid(visibility_logits)
        route_logits = F.avg_pool2d(logits.flatten(0, 2), self.routing_block, stride=self.routing_block)
        route_logits = route_logits.unflatten(0, (batch, steps, self.candidates))
        route_weights = route_logits.softmax(dim=2)
        choice = route_weights.argmax(dim=2)
        full_choice = choice.repeat_interleave(self.routing_block, 2).repeat_interleave(self.routing_block, 3)
        hard = F.one_hot(full_choice, num_classes=self.candidates).permute(0, 1, 4, 2, 3).to(parent.dtype)
        full_weights = route_weights.repeat_interleave(self.routing_block, 3).repeat_interleave(self.routing_block, 4)
        selection = hard + full_weights - full_weights.detach() if self.training else hard
        prediction = (selection[:, :, :, None] * candidates).sum(dim=2)
        return {
            "prediction": prediction,
            "candidates": candidates,
            "base_flow": base_flow,
            "refined_flow": refined_flow,
            "residual_flow": residual_flow,
            "route_logits": route_logits,
            "route_weights": route_weights,
            "visibility_logits": visibility_logits,
        }
