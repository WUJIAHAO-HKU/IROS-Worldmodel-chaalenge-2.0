"""Recurrent, action-pose-conditioned structure parent for Track 2.

The module is intentionally initialized as an exact identity on top of the
existing autoregressive parent.  It aligns the parent's RGB with a bounded
flow, predicts newly visible low-frequency structure, and restores only a
bounded high-frequency residual at native resolution.  Unlike the v22 display
refiner, its output is fed back into the next autoregressive step.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            _group_norm(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            _group_norm(channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.gelu(value + self.block(value))


class ActionOcclusionTextureParentV230(nn.Module):
    """A bounded recurrent extension of the frozen RGB parent.

    Three paths are deliberately separated:
      * residual flow moves structure already present in the parent frame;
      * synthesis predicts newly revealed coarse RGB structure;
      * a native-resolution path predicts high-frequency detail from all five
        real observations without downsampling the source texture first.
    """

    memory_frames = 5
    action_dim = 14
    pose_points = 3

    def __init__(
        self,
        maximum_flow_pixels: float = 6.0,
        maximum_structure_residual: float = 0.10,
        maximum_texture_residual: float = 0.05,
        initial_gate_logit: float = -4.0,
    ) -> None:
        super().__init__()
        self.maximum_flow_pixels = float(maximum_flow_pixels)
        self.maximum_structure_residual = float(maximum_structure_residual)
        self.maximum_texture_residual = float(maximum_texture_residual)
        self.initial_gate_logit = float(initial_gate_logit)

        # 21 visual channels + 17 broadcast action/horizon/arm channels.
        self.motion_stem = nn.Sequential(
            nn.Conv2d(38, 64, 3, padding=1),
            _group_norm(64),
            nn.GELU(),
            ResidualBlock(64, 1),
            ResidualBlock(64, 2),
            ResidualBlock(64, 4),
            nn.Conv2d(64, 32, 3, padding=1),
            _group_norm(32),
            nn.GELU(),
        )
        # flow xy, structure RGB, flow gate, structure gate.
        self.motion_head = nn.Conv2d(32, 7, 1)

        # warped/base/previous (12), five native RGB observations (15), their
        # high-pass maps (15), motion feature (16), and three pose maps = 61.
        self.texture_stem = nn.Sequential(
            nn.Conv2d(61, 32, 3, padding=1),
            _group_norm(32),
            nn.GELU(),
            ResidualBlock(32, 1),
            ResidualBlock(32, 2),
            nn.Conv2d(32, 24, 3, padding=1),
            _group_norm(24),
            nn.GELU(),
        )
        # texture RGB and texture visibility gate.
        self.texture_head = nn.Conv2d(24, 4, 1)

        # Zero residuals make an untrained v23 bit-exact with the frozen parent.
        nn.init.zeros_(self.motion_head.weight)
        nn.init.zeros_(self.motion_head.bias)
        nn.init.zeros_(self.texture_head.weight)
        nn.init.zeros_(self.texture_head.bias)
        with torch.no_grad():
            self.motion_head.bias[5:] = self.initial_gate_logit
            self.texture_head.bias[3] = self.initial_gate_logit

    @staticmethod
    def _highpass(value: torch.Tensor) -> torch.Tensor:
        return value - F.avg_pool2d(value, 5, 1, 2, count_include_pad=False)

    @staticmethod
    def _grid(batch: int, height: int, width: int, device, dtype) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, height, device=device, dtype=dtype),
            torch.linspace(-1, 1, width, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack((xx, yy), -1)[None].expand(batch, -1, -1, -1)

    @staticmethod
    def _pose_heatmaps(projected_pose: torch.Tensor, size: int = 64, sigma: float = 0.09) -> torch.Tensor:
        if projected_pose.ndim != 2 or projected_pose.shape[1] != 6:
            raise ValueError("projected_pose must be [B,6]")
        axis = torch.linspace(-1, 1, size, device=projected_pose.device, dtype=projected_pose.dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        points = projected_pose.reshape(len(projected_pose), 3, 2).clamp(-1.25, 1.25)
        distance = (xx[None, None] - points[:, :, 0, None, None]).square()
        distance = distance + (yy[None, None] - points[:, :, 1, None, None]).square()
        return torch.exp(-distance / (2 * sigma * sigma))

    def forward(
        self,
        base: torch.Tensor,
        previous: torch.Tensor,
        memory: torch.Tensor,
        action: torch.Tensor,
        projected_pose: torch.Tensor,
        horizon: torch.Tensor,
        arm: torch.Tensor,
    ):
        batch, channels, height, width = base.shape
        if channels != 3 or previous.shape != base.shape:
            raise ValueError("base and previous must be [B,3,H,W]")
        if memory.shape != (batch, 5, 3, height, width):
            raise ValueError("memory must be [B,5,3,H,W]")
        if action.shape != (batch, 14) or horizon.shape != (batch, 1) or arm.shape != (batch,):
            raise ValueError("action/horizon/arm input shape")

        base64 = F.interpolate(base, (64, 64), mode="bilinear", align_corners=False)
        previous64 = F.interpolate(previous, (64, 64), mode="bilinear", align_corners=False)
        memory_last64 = F.interpolate(memory[:, -1], (64, 64), mode="bilinear", align_corners=False)
        pose64 = self._pose_heatmaps(projected_pose)
        visual = torch.cat(
            (
                base64,
                previous64,
                base64 - previous64,
                (base64 - previous64).abs(),
                memory_last64,
                self._highpass(memory_last64),
                pose64,
            ),
            1,
        )
        arm_one_hot = F.one_hot(arm.long(), 2).to(base.dtype)
        condition = torch.cat((action, horizon, arm_one_hot), 1)
        condition = condition[:, :, None, None].expand(-1, -1, 64, 64)
        motion_feature = self.motion_stem(torch.cat((visual, condition), 1))
        raw = self.motion_head(motion_feature)
        flow_gate = raw[:, 5:6].sigmoid()
        structure_gate = raw[:, 6:7].sigmoid()
        flow64 = self.maximum_flow_pixels * torch.tanh(raw[:, :2]) * flow_gate
        flow = F.interpolate(flow64, (height, width), mode="bilinear", align_corners=False)

        grid = self._grid(batch, height, width, base.device, base.dtype)
        sample_grid = grid.clone()
        sample_grid[..., 0] += 2 * flow[:, 0] / max(width - 1, 1)
        sample_grid[..., 1] += 2 * flow[:, 1] / max(height - 1, 1)
        warped = F.grid_sample(base, sample_grid, mode="bilinear", padding_mode="border", align_corners=True)
        identity = F.grid_sample(base, grid, mode="bilinear", padding_mode="border", align_corners=True)
        # Parenthesize the interpolation residual: when flow is exactly zero,
        # warped and identity are identical and subtraction becomes exact before
        # adding it to the parent RGB tensor.
        aligned = base + (warped - identity)
        structure64 = self.maximum_structure_residual * torch.tanh(raw[:, 2:5]) * structure_gate
        structure = F.interpolate(structure64, (height, width), mode="bilinear", align_corners=False)
        coarse = aligned + structure

        memory_flat = memory.flatten(1, 2)
        memory_high = self._highpass(memory.flatten(0, 1)).reshape_as(memory).flatten(1, 2)
        detail_input = torch.cat(
            (
                coarse,
                base,
                previous,
                (coarse - previous).abs(),
                memory_flat,
                memory_high,
                F.interpolate(motion_feature[:, :16], (height, width), mode="bilinear", align_corners=False),
                F.interpolate(pose64, (height, width), mode="bilinear", align_corners=False),
            ),
            1,
        )
        texture_raw = self.texture_head(self.texture_stem(detail_input))
        texture_gate = texture_raw[:, 3:4].sigmoid()
        texture = self.maximum_texture_residual * torch.tanh(texture_raw[:, :3])
        texture = self._highpass(texture) * texture_gate
        output = (coarse + texture).clamp(0, 1)
        return output, {
            "aligned": aligned,
            "flow": flow,
            "flow_gate": flow_gate,
            "structure_gate": structure_gate,
            "texture_gate": texture_gate,
            "structure_residual": structure,
            "texture_residual": texture,
        }


def parameter_count() -> int:
    return sum(parameter.numel() for parameter in ActionOcclusionTextureParentV230().parameters())
