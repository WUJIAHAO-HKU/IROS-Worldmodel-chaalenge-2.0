"""Autoregressive U-Net with an explicit dark-structure generation head."""

from __future__ import annotations

import torch
from torch import nn

from .autoregressive_unet import OneStepActionUNet


class OneStepActionStructureUNet(OneStepActionUNet):
    """Generate RGB and an auxiliary robot/object silhouette from shared features."""

    def __init__(self) -> None:
        super().__init__()
        self.output = nn.Conv2d(self.base_channels, 4, kernel_size=3, padding=1)
        # This starts at exactly zero so loading an RGB parent is bit-exact.
        # Once trained, the predicted structure mask directly darkens the RGB
        # output instead of remaining an auxiliary signal that inference ignores.
        self.structure_strength = nn.Parameter(torch.zeros(()))

    def forward(self, context_frames: torch.Tensor, actions: torch.Tensor, return_structure: bool = False):
        batch, frames, channels, height, width = context_frames.shape
        if (frames, channels) != (self.context_frames, 3):
            raise ValueError("context_frames must be [B,5,3,H,W]")
        if actions.shape != (batch, self.conditioned_actions, self.action_dim):
            raise ValueError("actions must be [B,5,14]")
        e0 = self.enc0(context_frames.reshape(batch, frames * channels, height, width))
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        action = self.action_embedding(actions.flatten(1)).unsqueeze(-1).unsqueeze(-1)
        decoded = self.middle(torch.cat((e4, action.expand(-1, -1, *e4.shape[-2:])), dim=1))
        decoded = self.up3(decoded, e3)
        decoded = self.up2(decoded, e2)
        decoded = self.up1(decoded, e1)
        decoded = self.up0(decoded, e0)
        output = self.output(decoded)
        structure_logit = output[:, 3:4]
        # The repair is one-sided: it can restore missing dark structure but
        # cannot create the bright compensation blocks seen in earlier runs.
        structure_correction = self.structure_correction(structure_logit)
        prediction = context_frames[:, -1] + output[:, :3] - structure_correction
        return (prediction, structure_logit) if return_structure else prediction

    def structure_correction(self, structure_logit: torch.Tensor) -> torch.Tensor:
        strength = self.structure_strength.clamp(min=0.0, max=1.0)
        return 0.08 * strength * torch.sigmoid(structure_logit)


def load_rgb_parent(model: OneStepActionStructureUNet, state_dict: dict[str, torch.Tensor]) -> list[str]:
    """Load an RGB AR checkpoint and initialize the auxiliary mask head safely."""
    target = model.state_dict()
    copied = []
    for name, value in state_dict.items():
        if name == "output.weight":
            if value.shape != target[name][:3].shape:
                raise ValueError("parent RGB output weight shape does not match")
            target[name][:3].copy_(value)
            target[name][3:].zero_()
        elif name == "output.bias":
            if value.shape != target[name][:3].shape:
                raise ValueError("parent RGB output bias shape does not match")
            target[name][:3].copy_(value)
            target[name][3:].fill_(-2.0)
        elif name in target and target[name].shape == value.shape:
            target[name].copy_(value)
        else:
            raise ValueError(f"parent parameter does not match structure model: {name}")
        copied.append(name)
    model.load_state_dict(target, strict=True)
    return sorted(copied)


class OneStepActionSeparatedStructureUNet(OneStepActionUNet):
    """Bit-exact RGB parent plus an independent explicit missing-structure head."""

    def __init__(self) -> None:
        super().__init__()
        self.structure_output = nn.Conv2d(self.base_channels, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.structure_output.weight)
        nn.init.constant_(self.structure_output.bias, -2.0)
        self.structure_strength = nn.Parameter(torch.zeros(()))

    def structure_correction(self, structure_logit: torch.Tensor) -> torch.Tensor:
        strength = self.structure_strength.clamp(min=0.0, max=1.0)
        return 0.08 * strength * torch.sigmoid(structure_logit)

    def forward(self, context_frames: torch.Tensor, actions: torch.Tensor, return_structure: bool = False):
        batch, frames, channels, height, width = context_frames.shape
        if (frames, channels) != (self.context_frames, 3):
            raise ValueError("context_frames must be [B,5,3,H,W]")
        if actions.shape != (batch, self.conditioned_actions, self.action_dim):
            raise ValueError("actions must be [B,5,14]")
        e0 = self.enc0(context_frames.reshape(batch, frames * channels, height, width))
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        action = self.action_embedding(actions.flatten(1)).unsqueeze(-1).unsqueeze(-1)
        decoded = self.middle(torch.cat((e4, action.expand(-1, -1, *e4.shape[-2:])), dim=1))
        decoded = self.up3(decoded, e3)
        decoded = self.up2(decoded, e2)
        decoded = self.up1(decoded, e1)
        decoded = self.up0(decoded, e0)
        rgb = context_frames[:, -1] + self.output(decoded)
        structure_logit = self.structure_output(decoded)
        prediction = rgb - self.structure_correction(structure_logit)
        return (prediction, structure_logit) if return_structure else prediction


def load_separated_rgb_parent(model: OneStepActionSeparatedStructureUNet, state_dict: dict[str, torch.Tensor]) -> list[str]:
    incompatible = model.load_state_dict(state_dict, strict=False)
    expected_missing = {"structure_output.weight", "structure_output.bias", "structure_strength"}
    if incompatible.unexpected_keys or set(incompatible.missing_keys) != expected_missing:
        raise ValueError("parent parameter does not match separated structure model")
    return sorted(state_dict)
