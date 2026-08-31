"""One-step flow predictor recursively rolled out for the Track 2 contract."""

from __future__ import annotations

import torch

from .direct_flow_unet import DirectActionFlowUNet


class RecursiveActionFlowUNet(DirectActionFlowUNet):
    """Reuse Direct Flow weights one action at a time during an eight-step rollout.

    ``DirectActionFlowUNet`` conditions its horizon-zero decoder on the GRU state
    after four history actions and the first future action.  This class preserves
    that parameter layout, so a Direct Flow checkpoint is an exact warm start
    for its first recursive prediction rather than a loosely related initializer.
    """

    conditioning_actions = 5

    def forward(
        self,
        context_frames: torch.Tensor,
        actions: torch.Tensor,
        return_flow: bool = False,
    ):
        batch, frames, channels, height, width = context_frames.shape
        if (frames, channels) != (self.context_frames, 3):
            raise ValueError("context_frames must be [B,5,3,H,W]")
        if actions.shape != (batch, self.conditioning_actions, self.action_dim):
            raise ValueError("actions must be [B,5,14]")

        encoded = torch.cat(
            (
                context_frames.flatten(1, 2),
                context_frames[:, -1] - context_frames[:, -2],
                context_frames[:, -1] - context_frames[:, 0],
                self._coordinates(batch, height, width, context_frames.device, context_frames.dtype),
            ),
            dim=1,
        )
        e0 = self.enc0(encoded)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        states, _ = self.action_gru(actions)
        condition = self.action_norm(states[:, -1])
        decoded = self._modulate(self.middle(e4), condition, self.film[0])
        decoded = self._modulate(self.up3(decoded, e3), condition, self.film[1])
        decoded = self._modulate(self.up2(decoded, e2), condition, self.film[2])
        decoded = self._modulate(self.up1(decoded, e1), condition, self.film[3])
        decoded = self._modulate(self.up0(decoded, e0), condition, self.film[4])
        output = self.output(decoded)
        flow = torch.tanh(output[:, :2]) * self.max_flow_pixels
        correction = torch.tanh(output[:, 2:5]) * 0.35
        blend = torch.sigmoid(output[:, 5:6])
        warped = self._warp(context_frames[:, -1], flow[:, None])[:, 0]
        prediction = blend * warped + (1.0 - blend) * context_frames[:, -1] + correction
        return (prediction, flow) if return_flow else prediction
