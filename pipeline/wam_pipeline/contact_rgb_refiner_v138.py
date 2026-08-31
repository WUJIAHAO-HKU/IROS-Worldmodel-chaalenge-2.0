"""Action-conditioned recurrent warp/fusion refiner for the contact crop."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .contact_occlusion_head_v13 import ConvGRUCell


class Block(nn.Module):
    def __init__(self, inputs: int, outputs: int, stride: int = 1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inputs, outputs, 3 if stride == 1 else 4, stride=stride,
                      padding=1), nn.GroupNorm(8, outputs), nn.SiLU(),
            nn.Conv2d(outputs, outputs, 3, padding=1),
            nn.GroupNorm(8, outputs), nn.SiLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor: return self.layers(value)


class ContactRGBRefinerV138(nn.Module):
    """Warp observed/parent material while recurrently predicting new geometry."""

    def __init__(self, base_channels: int = 48, maximum_flow: float = 28.0,
                 maximum_correction: float = 0.10) -> None:
        super().__init__(); c = base_channels
        if c % 8: raise ValueError("base_channels must be divisible by 8")
        self.base_channels = c; self.maximum_flow = maximum_flow
        self.maximum_correction = maximum_correction
        # last/current/delta RGB (9), source/parent bottle-black-grey (6), xy (2).
        self.enc0 = Block(17, c)
        self.enc1 = Block(c, 2 * c, stride=2)
        self.enc2 = Block(2 * c, 3 * c, stride=2)
        self.action = nn.Sequential(nn.Linear(7, c), nn.SiLU(), nn.Linear(c, c), nn.SiLU())
        self.arm = nn.Embedding(2, c)
        self.recurrent = ConvGRUCell(5 * c, 3 * c)
        self.middle = Block(6 * c, 3 * c)
        self.up1 = Block(5 * c, 2 * c)
        self.up0 = Block(3 * c, c)
        # parent flow, observed-source flow, 3 source logits, RGB correction.
        self.output = nn.Conv2d(c, 10, 3, padding=1)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
        with torch.no_grad(): self.output.bias[4] = 3.0  # identity parent candidate

    @staticmethod
    def coordinates(batch: int, height: int, width: int, value: torch.Tensor) -> torch.Tensor:
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=value.device, dtype=value.dtype),
            torch.linspace(-1, 1, width, device=value.device, dtype=value.dtype), indexing="ij"
        )
        return torch.stack((x, y))[None].expand(batch, -1, -1, -1)

    @staticmethod
    def warp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = image.shape
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=image.device, dtype=image.dtype),
            torch.linspace(-1, 1, width, device=image.device, dtype=image.dtype), indexing="ij"
        )
        base = torch.stack((x, y), dim=-1)[None].expand(batch, -1, -1, -1)
        scale = image.new_tensor((2 / max(width - 1, 1), 2 / max(height - 1, 1)))
        grid = base + flow.permute(0, 2, 3, 1) * scale
        return F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def forward(self, last: torch.Tensor, parent: torch.Tensor, actions: torch.Tensor,
                arm_id: torch.Tensor, source_semantic: torch.Tensor,
                parent_semantic: torch.Tensor, return_details: bool = False):
        batch, steps, _, height, width = parent.shape
        hidden = None; previous = last; predictions = []; flows = []; weights = []; corrections = []
        arm = self.arm(arm_id)[:, :, None, None]
        xy = self.coordinates(batch, height, width, last)
        for time in range(steps):
            current = parent[:, time]
            encoded = torch.cat((last, current, current - previous, source_semantic,
                                 parent_semantic[:, time], xy), dim=1)
            e0 = self.enc0(encoded); e1 = self.enc1(e0); e2 = self.enc2(e1)
            action = self.action(actions[:, time])[:, :, None, None].expand(-1, -1, *e2.shape[-2:])
            arm_feature = arm.expand(-1, -1, *e2.shape[-2:])
            hidden = self.recurrent(torch.cat((e2, action, arm_feature), dim=1), hidden)
            middle = self.middle(torch.cat((e2, hidden), dim=1))
            decoded = F.interpolate(middle, e1.shape[-2:], mode="bilinear", align_corners=False)
            decoded = self.up1(torch.cat((decoded, e1), dim=1))
            decoded = F.interpolate(decoded, e0.shape[-2:], mode="bilinear", align_corners=False)
            decoded = self.up0(torch.cat((decoded, e0), dim=1))
            raw = self.output(decoded)
            flow = torch.tanh(raw[:, :4]).reshape(batch, 2, 2, height, width) * self.maximum_flow
            source_weights = torch.softmax(raw[:, 4:7], dim=1)
            correction = torch.tanh(raw[:, 7:]) * self.maximum_correction
            candidates = torch.stack((current, self.warp(current, flow[:, 0]),
                                      self.warp(last, flow[:, 1])), dim=1)
            prediction = (source_weights[:, :, None] * candidates).sum(dim=1) + correction
            predictions.append(prediction); flows.append(flow); weights.append(source_weights)
            corrections.append(correction); previous = current
        prediction = torch.stack(predictions, dim=1)
        if not return_details: return prediction
        return {"prediction": prediction, "flow": torch.stack(flows, dim=1),
                "source_weight": torch.stack(weights, dim=1),
                "correction": torch.stack(corrections, dim=1)}
