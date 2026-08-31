"""Direct eight-horizon RGB predictor initialized from the one-step parent."""

from __future__ import annotations

import torch
from torch import nn

from .residual_unet import ConvBlock, DownBlock, UpBlock


class DirectHorizonActionUNet(nn.Module):
    """Predict all future residuals at once, avoiding autoregressive drift."""

    context_frames = 5
    history_actions = 4
    prediction_frames = 8
    action_dim = 14
    base_channels = 48

    def __init__(self) -> None:
        super().__init__()
        base = self.base_channels
        self.enc0 = ConvBlock(15, base)
        self.enc1 = DownBlock(base, 96)
        self.enc2 = DownBlock(96, 144)
        self.enc3 = DownBlock(144, 192)
        self.enc4 = DownBlock(192, 256)
        self.action_embedding = nn.Sequential(
            nn.Linear((self.history_actions + self.prediction_frames) * self.action_dim, 512),
            nn.SiLU(), nn.Linear(512, 256), nn.SiLU(),
        )
        self.middle = ConvBlock(512, 256)
        self.up3 = UpBlock(256, 192, 192)
        self.up2 = UpBlock(192, 144, 144)
        self.up1 = UpBlock(144, 96, 96)
        self.up0 = UpBlock(96, base, base)
        self.output = nn.Conv2d(base, self.prediction_frames * 3, 3, padding=1)

    def forward(self, context: torch.Tensor, history: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        batch, _, _, height, width = context.shape
        e0 = self.enc0(context.flatten(1, 2)); e1 = self.enc1(e0); e2 = self.enc2(e1); e3 = self.enc3(e2); e4 = self.enc4(e3)
        actions = torch.cat((history, future), dim=1).flatten(1)
        action = self.action_embedding(actions)[:, :, None, None].expand(-1, -1, *e4.shape[-2:])
        x = self.middle(torch.cat((e4, action), dim=1))
        x = self.up3(x, e3); x = self.up2(x, e2); x = self.up1(x, e1); x = self.up0(x, e0)
        residual = self.output(x).unflatten(1, (self.prediction_frames, 3))
        return context[:, -1:, :, :, :] + residual


def load_one_step_parent(model: DirectHorizonActionUNet, state: dict[str, torch.Tensor]) -> list[str]:
    """Expand a trained one-step parent into the direct multi-horizon model."""
    target = model.state_dict()
    copied = []
    for name, value in state.items():
        if name == "action_embedding.0.weight":
            # Four history actions retain their exact parent weights. Each of
            # the eight future actions initially contributes an equal share of
            # the parent's current-action conditioning.
            target[name].zero_()
            target[name][:, :56].copy_(value[:, :56])
            for horizon in range(8):
                start = 56 + 14 * horizon
                target[name][:, start:start + 14].copy_(value[:, 56:70] / 8.0)
        elif name == "output.weight":
            target[name].copy_(value.repeat(8, 1, 1, 1))
        elif name == "output.bias":
            target[name].copy_(value.repeat(8))
        elif name in target and target[name].shape == value.shape:
            target[name].copy_(value)
        else:
            raise ValueError(f"cannot expand parent parameter {name}: {tuple(value.shape)}")
        copied.append(name)
    model.load_state_dict(target, strict=True)
    return copied
