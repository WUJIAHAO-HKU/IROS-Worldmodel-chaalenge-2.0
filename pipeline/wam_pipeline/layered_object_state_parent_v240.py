"""Object-layered recurrent parent extension for the Track-2 v24 experiment.

The module never asks a decoder to redraw lettering or gripper texture.  RGB
comes from a rigidly transported real observation.  A recurrent low-resolution
state predicts only residual geometry and visibility for beam, black gripper,
grey gripper and bottle layers; native-resolution hard compositing preserves
the observed pixels and keeps the frozen parent everywhere else.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


LAYER_COUNT = 4


def _norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            _norm(channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1), _norm(channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.silu(value + self.body(value))


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        total = input_channels + hidden_channels
        self.gates = nn.Conv2d(total, 2 * hidden_channels, 3, padding=1)
        self.candidate = nn.Conv2d(total, hidden_channels, 3, padding=1)

    def forward(self, value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        reset, update = self.gates(torch.cat((value, state), 1)).sigmoid().chunk(2, 1)
        candidate = torch.tanh(self.candidate(torch.cat((value, reset * state), 1)))
        return update * state + (1 - update) * candidate


class LayeredObjectStateParentV240(nn.Module):
    """Eight-step explicit layer state with native-resolution texture transport."""

    input_channels = 34
    layer_order = (0, 3, 2, 1)  # beam, bottle, grey jaw, black pad

    def __init__(self, channels: int = 48, maximum_residual_flow: float = 8.0) -> None:
        super().__init__()
        self.channels = int(channels)
        self.maximum_residual_flow = float(maximum_residual_flow)
        self.stem = nn.Sequential(
            nn.Conv2d(self.input_channels, channels, 3, padding=1),
            _norm(channels), nn.SiLU(), ResidualBlock(channels, 1),
            ResidualBlock(channels, 2),
        )
        self.recurrent = ConvGRUCell(channels, channels)
        self.decode = nn.Sequential(ResidualBlock(channels, 1), ResidualBlock(channels, 4))
        # Four residual xy flows, four visibility logits and one stale cleanup gate.
        self.head = nn.Conv2d(channels, 13, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[8:12] = -4.0
            self.head.bias[12] = -6.0

    @staticmethod
    def pose_heatmaps(projected_pose: torch.Tensor, size: int = 64,
                      sigma: float = 0.09) -> torch.Tensor:
        points = projected_pose.reshape(len(projected_pose), 3, 2).clamp(-1.25, 1.25)
        axis = torch.linspace(-1, 1, size, device=projected_pose.device,
                              dtype=projected_pose.dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        distance = (xx[None, None] - points[:, :, 0, None, None]).square()
        distance += (yy[None, None] - points[:, :, 1, None, None]).square()
        return torch.exp(-distance / (2 * sigma * sigma))

    @staticmethod
    def _grid(batch: int, height: int, width: int, device, dtype) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, height, device=device, dtype=dtype),
            torch.linspace(-1, 1, width, device=device, dtype=dtype), indexing="ij",
        )
        return torch.stack((xx, yy), -1)[None].expand(batch, -1, -1, -1)

    @staticmethod
    def _straight_through_mask(probability: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        hard = (probability >= float(threshold)).to(probability.dtype)
        return hard + probability - probability.detach()

    def _render_frame(
        self,
        base: torch.Tensor,
        transported: torch.Tensor,
        support: torch.Tensor,
        cleanup: torch.Tensor,
        cleanup_support: torch.Tensor,
        raw: torch.Tensor,
        mask_threshold: float,
        enabled_layers: tuple[bool, ...],
        frame_active: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, _, height, width = base.shape
        flow64 = self.maximum_residual_flow * torch.tanh(raw[:, :8]).reshape(
            batch, LAYER_COUNT, 2, 64, 64
        )
        flow = F.interpolate(flow64.flatten(0, 1), (height, width), mode="bilinear",
                             align_corners=False).unflatten(0, (batch, LAYER_COUNT))
        visibility = raw[:, 8:12].sigmoid()
        cleanup_probability = raw[:, 12:13].sigmoid()
        grid = self._grid(batch, height, width, base.device, base.dtype)
        layer_rgbs, layer_masks, soft_masks, warped_supports = [], [], [], []
        for layer in range(LAYER_COUNT):
            sample = grid.clone()
            sample[..., 0] += 2 * flow[:, layer, 0] / max(width - 1, 1)
            sample[..., 1] += 2 * flow[:, layer, 1] / max(height - 1, 1)
            rgb = F.grid_sample(transported, sample, mode="bilinear",
                                padding_mode="border", align_corners=True)
            layer_support = F.grid_sample(support[:, layer:layer + 1], sample,
                                          mode="bilinear", padding_mode="zeros",
                                          align_corners=True)
            visible = F.interpolate(visibility[:, layer:layer + 1], (height, width),
                                    mode="bilinear", align_corners=False)
            probability = layer_support * visible
            if not frame_active or not enabled_layers[layer]:
                probability = probability * 0
            layer_rgbs.append(rgb); warped_supports.append(layer_support)
            soft_masks.append(probability)
            layer_masks.append(self._straight_through_mask(probability, mask_threshold))
        masks = torch.stack(layer_masks, 1)
        cleanup_alpha = F.interpolate(cleanup_probability, (height, width), mode="bilinear",
                                      align_corners=False) * cleanup_support
        if not frame_active:
            cleanup_alpha = cleanup_alpha * 0
        cleanup_alpha = self._straight_through_mask(cleanup_alpha, mask_threshold)
        output = cleanup_alpha * cleanup + (1 - cleanup_alpha) * base
        for layer in self.layer_order:
            alpha = masks[:, layer]
            output = alpha * layer_rgbs[layer] + (1 - alpha) * output
        return (output.clamp(0, 1), masks[:, :, 0],
                torch.stack(soft_masks, 1)[:, :, 0],
                torch.stack(warped_supports, 1)[:, :, 0],
                torch.stack(layer_rgbs, 1), flow, visibility)

    def forward(
        self,
        base: torch.Tensor,
        transported: torch.Tensor,
        support: torch.Tensor,
        cleanup: torch.Tensor,
        cleanup_support: torch.Tensor,
        actions: torch.Tensor,
        projected_pose: torch.Tensor,
        arms: torch.Tensor,
        target_masks: torch.Tensor | None = None,
        teacher_forcing: float = 0.0,
        mask_threshold: float = 0.5,
        active_from: int = 0,
        enabled_layers: tuple[bool, ...] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, time, channels, height, width = base.shape
        expected_rgb = (batch, time, 3, height, width)
        if base.shape != expected_rgb or transported.shape != expected_rgb:
            raise ValueError("base and transported must be [B,T,3,H,W]")
        if support.shape != (batch, time, LAYER_COUNT, height, width):
            raise ValueError("support must be [B,T,4,H,W]")
        if actions.shape != (batch, time, 7) or projected_pose.shape != (batch, time, 6):
            raise ValueError("actions/pose shapes differ from rollout")
        state = base.new_zeros((batch, self.channels, 64, 64))
        if enabled_layers is None:
            enabled_layers = (True,) * LAYER_COUNT
        if len(enabled_layers) != LAYER_COUNT:
            raise ValueError("enabled_layers must contain four booleans")
        previous_masks = F.interpolate(support[:, 0], (64, 64), mode="nearest")
        outputs, masks, soft_masks, warped_supports = [], [], [], []
        layer_rgbs, flows, visibilities = [], [], []
        for index in range(time):
            values = [base[:, index], transported[:, index],
                      (transported[:, index] - base[:, index]).abs(), support[:, index],
                      cleanup[:, index] - base[:, index], cleanup_support[:, index],
                      F.interpolate(previous_masks, (height, width), mode="nearest")]
            visual = F.interpolate(torch.cat(values, 1), (64, 64), mode="bilinear",
                                   align_corners=False)
            horizon = base.new_full((batch, 1), (index + 1) / time)
            arm_one_hot = F.one_hot(arms.long(), 2).to(base.dtype)
            condition = torch.cat((actions[:, index], horizon, arm_one_hot), 1)
            condition = condition[:, :, None, None].expand(-1, -1, 64, 64)
            pose = self.pose_heatmaps(projected_pose[:, index])
            encoded = self.stem(torch.cat((visual, condition, pose), 1))
            state = self.recurrent(encoded, state)
            raw = self.head(self.decode(state))
            output, mask, soft_mask, warped_support, layer_rgb, flow, visibility = self._render_frame(
                base[:, index], transported[:, index], support[:, index], cleanup[:, index],
                cleanup_support[:, index], raw, mask_threshold, enabled_layers,
                index >= int(active_from),
            )
            outputs.append(output); masks.append(mask); soft_masks.append(soft_mask)
            warped_supports.append(warped_support); layer_rgbs.append(layer_rgb); flows.append(flow)
            visibilities.append(visibility)
            predicted = F.interpolate(mask, (64, 64), mode="area")
            if target_masks is not None and teacher_forcing > 0:
                truth = F.interpolate(target_masks[:, index], (64, 64), mode="nearest")
                previous_masks = teacher_forcing * truth + (1 - teacher_forcing) * predicted
            else:
                previous_masks = predicted
        return torch.stack(outputs, 1), {
            "masks": torch.stack(masks, 1),
            "soft_masks": torch.stack(soft_masks, 1),
            "warped_support": torch.stack(warped_supports, 1),
            "layer_rgbs": torch.stack(layer_rgbs, 1),
            "flows": torch.stack(flows, 1),
            "visibility": torch.stack(visibilities, 1),
        }


def parameter_count(channels: int = 48) -> int:
    return sum(parameter.numel() for parameter in LayeredObjectStateParentV240(channels).parameters())
