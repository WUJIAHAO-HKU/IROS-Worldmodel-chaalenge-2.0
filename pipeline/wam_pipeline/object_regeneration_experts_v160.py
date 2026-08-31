"""Object-level glyph and gripper regeneration experts for the v16 experiment.

The frozen video parent is never asked to invent high-frequency appearance in
this layer.  Glyph pixels are sampled from a train-only sharp atlas, while the
contact expert predicts black/grey instances, dense transport, visibility and
a bounded replacement image.  Both renderers expose an edit alpha so an
episode-disjoint router can always fall back to the parent pixel-for-pixel.
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
        return hidden + update * (candidate - hidden)


def _block(input_channels: int, output_channels: int, stride: int = 1) -> nn.Sequential:
    groups = max(min(output_channels // 8, 8), 1)
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1),
        nn.GroupNorm(groups, output_channels), nn.SiLU(),
        nn.Conv2d(output_channels, output_channels, 3, padding=1),
        nn.GroupNorm(groups, output_channels), nn.SiLU(),
    )


class RecurrentObjectUNet(nn.Module):
    """Native-resolution encoder/ConvGRU/decoder shared by both experts."""

    def __init__(self, input_channels: int, output_channels: int, base_channels: int = 24) -> None:
        super().__init__()
        c = base_channels
        self.enc0 = _block(input_channels, c)
        self.enc1 = _block(c, 2 * c, 2)
        self.enc2 = _block(2 * c, 4 * c, 2)
        self.recurrent = ConvGRUCell(4 * c + 2 * c, 4 * c)
        self.action = nn.Sequential(nn.Linear(7, 2 * c), nn.SiLU(), nn.Linear(2 * c, 2 * c))
        self.arm = nn.Embedding(2, 2 * c)
        self.dec1 = _block(4 * c + 2 * c, 2 * c)
        self.dec0 = _block(2 * c + c, c)
        self.head = nn.Conv2d(c, output_channels, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward_step(self, visual: torch.Tensor, action: torch.Tensor,
                     arm_id: torch.Tensor, hidden: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        e0 = self.enc0(visual)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        condition = self.action(action) + self.arm(arm_id)
        condition = condition[:, :, None, None].expand(-1, -1, *e2.shape[-2:])
        hidden = self.recurrent(torch.cat((e2, condition), 1), hidden)
        d1 = F.interpolate(hidden, e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat((d1, e1), 1))
        d0 = F.interpolate(d1, e0.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.dec0(torch.cat((d0, e0), 1))), hidden


def dense_warp(value: torch.Tensor, flow: torch.Tensor,
               mode: str = "bilinear") -> torch.Tensor:
    """Sample a static source into every target frame using normalized flow."""
    batch, time, _, height, width = flow.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, height, device=value.device, dtype=value.dtype),
        torch.linspace(-1, 1, width, device=value.device, dtype=value.dtype), indexing="ij"
    )
    grid = torch.stack((xx, yy), -1)[None, None] + flow.permute(0, 1, 3, 4, 2)
    source = value[:, None].expand(-1, time, -1, -1, -1).flatten(0, 1)
    warped = F.grid_sample(source, grid.flatten(0, 1), mode=mode,
                           padding_mode="border" if mode == "bilinear" else "zeros",
                           align_corners=True)
    return warped.unflatten(0, (batch, time))


class GlyphAtlasProjectionExpert(nn.Module):
    """Predict dense atlas-to-future geometry and per-pixel visibility."""

    def __init__(self, base_channels: int = 24, maximum_flow: float = 0.28) -> None:
        super().__init__()
        self.maximum_flow = maximum_flow
        # atlas/current/delta RGB + atlas mask/current beam/current glyph = 12.
        self.network = RecurrentObjectUNet(12, 4, base_channels)

    def forward(self, atlas_rgb: torch.Tensor, parent: torch.Tensor,
                actions: torch.Tensor, arm_id: torch.Tensor,
                atlas_mask: torch.Tensor, parent_beam: torch.Tensor,
                parent_glyph: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = None
        previous = atlas_rgb
        flows, visibility, edit = [], [], []
        for time in range(parent.shape[1]):
            current = parent[:, time]
            visual = torch.cat((atlas_rgb, current, current - previous, atlas_mask,
                                parent_beam[:, time], parent_glyph[:, time]), 1)
            raw, hidden = self.network.forward_step(visual, actions[:, time], arm_id, hidden)
            flows.append(self.maximum_flow * torch.tanh(raw[:, :2]))
            visibility.append(raw[:, 2:3])
            edit.append(raw[:, 3:4])
            previous = current
        return torch.stack(flows, 1), torch.stack(visibility, 1), torch.stack(edit, 1)


def render_atlas_glyph(parent: torch.Tensor, atlas_rgb: torch.Tensor,
                       atlas_mask: torch.Tensor, parent_beam: torch.Tensor,
                       flow: torch.Tensor, visibility_logits: torch.Tensor,
                       edit_logits: torch.Tensor, strength: float = 1.0,
                       parent_clean: torch.Tensor | None = None,
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    warped_rgb = dense_warp(atlas_rgb, flow)
    warped_mask = dense_warp(atlas_mask, flow).clamp(0, 1)
    # A small dilation makes the renderer robust to one-pixel beam-mask errors
    # without allowing lettering to leak into the white background.
    beam = F.max_pool3d(parent_beam, (1, 5, 5), 1, (0, 2, 2))
    edit_probability = visibility_logits.sigmoid() * edit_logits.sigmoid()
    alpha = (strength * warped_mask * edit_probability * beam).clamp(0, 1)
    # Cleaning is itself routed by the predicted atlas support.  Earlier
    # versions always returned the inpainted parent outside alpha and could
    # erase a correct word even when the expert had near-zero confidence.
    if parent_clean is None:
        parent_clean = parent
    cleanup = (strength * F.max_pool3d(warped_mask * edit_probability, (1, 5, 5), 1, (0, 2, 2))
               * beam).clamp(0, 1)
    base = parent * (1 - cleanup) + parent_clean * cleanup
    output = base * (1 - alpha) + warped_rgb * alpha
    return output, alpha, warped_mask


class GripperInstanceFlowExpert(nn.Module):
    """Regenerate black/grey contact instances and explicitly clear stale pixels."""

    def __init__(self, base_channels: int = 24, maximum_flow: float = 0.32) -> None:
        super().__init__()
        self.maximum_flow = maximum_flow
        # source/current/delta RGB + source black/grey + parent black/grey/bottle = 14.
        # output: semantic residuals(3), flow(2), visibility, RGB replacement(3), edit.
        self.network = RecurrentObjectUNet(14, 10, base_channels)

    def forward(self, source: torch.Tensor, parent: torch.Tensor,
                actions: torch.Tensor, arm_id: torch.Tensor,
                source_black: torch.Tensor, source_grey: torch.Tensor,
                parent_black: torch.Tensor, parent_grey: torch.Tensor,
                parent_bottle: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = None
        previous = source
        semantics, flows, visibility, replacement, edit = [], [], [], [], []
        for time in range(parent.shape[1]):
            current = parent[:, time]
            visual = torch.cat((source, current, current - previous, source_black, source_grey,
                                parent_black[:, time], parent_grey[:, time], parent_bottle[:, time]), 1)
            raw, hidden = self.network.forward_step(visual, actions[:, time], arm_id, hidden)
            prior = torch.cat((1 - torch.maximum(parent_black[:, time], parent_grey[:, time]),
                               parent_black[:, time], parent_grey[:, time]), 1).clamp(0.02, 0.98)
            semantics.append(prior.log() + raw[:, :3])
            flows.append(self.maximum_flow * torch.tanh(raw[:, 3:5]))
            visibility.append(raw[:, 5:6])
            replacement.append(raw[:, 6:9])
            edit.append(raw[:, 9:10])
            previous = current
        return (torch.stack(semantics, 1), torch.stack(flows, 1),
                torch.stack(visibility, 1), torch.stack(replacement, 1), torch.stack(edit, 1))


def render_gripper_instances(parent: torch.Tensor, source: torch.Tensor,
                             source_black: torch.Tensor, source_grey: torch.Tensor,
                             parent_black: torch.Tensor, parent_grey: torch.Tensor,
                             semantic_logits: torch.Tensor, flow: torch.Tensor,
                             visibility_logits: torch.Tensor, replacement_delta: torch.Tensor,
                             edit_logits: torch.Tensor, strength: float = 1.0
                             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    probability = semantic_logits.softmax(2)
    warped_rgb = dense_warp(source, flow)
    warped_structure = dense_warp(torch.maximum(source_black, source_grey), flow).clamp(0, 1)
    predicted_structure = probability[:, :, 1:].sum(2, keepdim=True)
    stale = torch.maximum(parent_black, parent_grey)
    union = torch.maximum(torch.maximum(predicted_structure, warped_structure), stale)
    support = F.max_pool3d(union, (1, 9, 9), 1, (0, 4, 4))
    visibility = visibility_logits.sigmoid()
    transported = visibility * warped_rgb + (1 - visibility) * parent
    # The learned replacement can brighten stale black pixels as well as paint
    # missing structure, which a residual restricted to predicted black cannot.
    candidate = (transported + 0.75 * torch.tanh(replacement_delta)).clamp(0, 1)
    alpha = (strength * support * edit_logits.sigmoid()).clamp(0, 1)
    output = parent * (1 - alpha) + candidate * alpha
    return output, probability, alpha, warped_structure


def parameter_counts(base_channels: int = 24) -> dict[str, int]:
    glyph = GlyphAtlasProjectionExpert(base_channels)
    gripper = GripperInstanceFlowExpert(base_channels)
    return {
        "glyph_atlas": sum(value.numel() for value in glyph.parameters()),
        "gripper_instance_flow": sum(value.numel() for value in gripper.parameters()),
    }
