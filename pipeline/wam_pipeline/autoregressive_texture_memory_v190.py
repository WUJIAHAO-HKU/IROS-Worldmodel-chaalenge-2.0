"""Autoregressive parent with persistent native-resolution texture memory."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .autoregressive_unet import OneStepActionUNet
from .residual_unet import ConvBlock


class OneStepActionTextureMemoryUNet(OneStepActionUNet):
    """Keep the five real observations available throughout all eight steps.

    The inherited U-Net still predicts dynamics.  A new internal branch warps
    one of the original native-resolution frames and replaces only high-frequency
    detail (plus a bounded low-frequency structure correction) before the next
    autoregressive step.  Source choice is hard in the forward pass so letters
    are never averaged across observations.
    """

    memory_sources = 5

    def __init__(self, maximum_flow: float = .60, structure_scale: float = .35, use_source_expert: bool = False) -> None:
        super().__init__(); self.maximum_flow = maximum_flow; self.structure_scale = structure_scale;self.use_source_expert=use_source_expert
        self.memory_enc0 = ConvBlock(15, self.base_channels)
        self.memory_fusion = ConvBlock(self.base_channels * 2 + 3, 64)
        # 5x flow, five source scores, texture gate, structure gate.
        self.memory_head = nn.Conv2d(64, 17, 3, padding=1)
        self.benefit_head = nn.Conv2d(64, 2, 3, padding=1)
        self.benefit_head_right = nn.Conv2d(64, 2, 3, padding=1)
        def benefit_expert():
            return nn.Sequential(nn.Conv2d(78,48,3,padding=1),nn.GELU(),
                nn.Conv2d(48,48,3,padding=2,dilation=2),nn.GELU(),
                nn.Conv2d(48,32,3,padding=4,dilation=4),nn.GELU(),nn.Conv2d(32,2,3,padding=1))
        self.benefit_expert = benefit_expert();self.benefit_expert_right = benefit_expert()
        self.source_expert=nn.Sequential(nn.Conv2d(97,64,3,padding=1),nn.GELU(),
            nn.Conv2d(64,64,3,padding=2,dilation=2),nn.GELU(),
            nn.Conv2d(64,32,3,padding=4,dilation=4),nn.GELU(),nn.Conv2d(32,5,3,padding=1))
        nn.init.zeros_(self.memory_head.weight); nn.init.zeros_(self.memory_head.bias)
        nn.init.zeros_(self.benefit_head.weight); nn.init.constant_(self.benefit_head.bias,-6.)
        nn.init.zeros_(self.benefit_head_right.weight); nn.init.constant_(self.benefit_head_right.bias,-6.)
        for expert in (self.benefit_expert,self.benefit_expert_right):
            nn.init.zeros_(expert[-1].weight);nn.init.constant_(expert[-1].bias,-6.)
        nn.init.zeros_(self.source_expert[-1].weight);nn.init.zeros_(self.source_expert[-1].bias)
        with torch.no_grad():
            self.memory_head.bias[10:15] = torch.tensor([-2., -1., 0., 1., 3.])
            self.memory_head.bias[15:] = -6.0

    def initialize_memory_encoder(self) -> None:
        """Copy the already-trained full-resolution visual encoder."""
        self.memory_enc0.load_state_dict(self.enc0.state_dict(), strict=True)

    @staticmethod
    def _base_grid(batch: int, height: int, width: int, device, dtype) -> torch.Tensor:
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, height, device=device, dtype=dtype),
                                torch.linspace(-1, 1, width, device=device, dtype=dtype), indexing="ij")
        return torch.stack((xx, yy), -1)[None].expand(batch, -1, -1, -1)

    def forward(self, context_frames: torch.Tensor, actions: torch.Tensor,
                memory_frames: torch.Tensor | None = None,
                return_diagnostics: bool = False):
        batch, frames, channels, height, width = context_frames.shape
        if memory_frames is None: memory_frames = context_frames
        if memory_frames.shape != context_frames.shape:
            raise ValueError("memory_frames must match context_frames")
        e0 = self.enc0(context_frames.reshape(batch, frames * channels, height, width))
        e1 = self.enc1(e0); e2 = self.enc2(e1); e3 = self.enc3(e2); e4 = self.enc4(e3)
        action = self.action_embedding(actions.flatten(1)).unsqueeze(-1).unsqueeze(-1)
        decoded = self.middle(torch.cat((e4, action.expand(-1, -1, *e4.shape[-2:])), 1))
        decoded = self.up3(decoded, e3); decoded = self.up2(decoded, e2)
        decoded = self.up1(decoded, e1); decoded = self.up0(decoded, e0)
        base = context_frames[:, -1] + self.output(decoded)

        memory_feature = self.memory_enc0(memory_frames.reshape(batch, 15, height, width))
        fused = self.memory_fusion(torch.cat((decoded, memory_feature, base), 1))
        raw = self.memory_head(fused)
        action_delta=(actions[:,-1]-actions[:,-2]).abs()
        right_active=(action_delta[:,7:].mean(1)>action_delta[:,:7].mean(1))[:,None,None,None]
        flow = self.maximum_flow * torch.tanh(raw[:, :10].reshape(batch, 5, 2, height, width))
        grid = self._base_grid(batch, height, width, base.device, base.dtype)
        warped = []
        for source in range(5):
            warped.append(F.grid_sample(memory_frames[:, source], grid + flow[:, source].permute(0, 2, 3, 1),
                                        mode="bilinear", padding_mode="border", align_corners=True))
        warped = torch.stack(warped, 1)
        source_input=torch.cat((fused,base,warped.flatten(1,2),(warped-base[:,None]).abs().flatten(1,2)),1)
        source_logits=self.source_expert(source_input.float()) if self.use_source_expert else raw[:,10:15]
        soft = source_logits.softmax(1)
        hard = F.one_hot(soft.argmax(1), 5).permute(0, 3, 1, 2).to(soft.dtype)
        source_weight = hard.detach() - soft.detach() + soft
        selected = (warped * source_weight[:, :, None]).sum(1)
        highpass = lambda value: value - F.avg_pool2d(value, 5, 1, 2, count_include_pad=False)
        confidence=soft.max(1,keepdim=True).values;flow_magnitude=flow.square().sum(2).sqrt().mean(1,keepdim=True)
        gate_input=torch.cat((fused,base,selected,(selected-base).abs(),(highpass(selected)-highpass(base)).abs(),confidence,flow_magnitude),1)
        gate_input=torch.nan_to_num(gate_input.float(),nan=0.,posinf=1.,neginf=-1.).clamp(-4,4)
        with torch.autocast(device_type=gate_input.device.type,enabled=False):
            benefit_raw=torch.where(right_active,self.benefit_expert_right(gate_input),self.benefit_expert(gate_input)).clamp(-12,12)
        texture_gate = benefit_raw[:, :1].sigmoid()
        structure_gate = benefit_raw[:, 1:2].sigmoid()
        output = base + texture_gate * (highpass(selected) - highpass(base))
        output = output + self.structure_scale * structure_gate * (selected - base)
        if return_diagnostics:
            return output, {"base": base, "selected": selected, "flow": flow,
                            "warped": warped, "source_logits": source_logits,
                            "texture_logit": benefit_raw[:, :1], "structure_logit": benefit_raw[:, 1:2],
                            "right_active": right_active,
                            "source_weight": source_weight, "texture_gate": texture_gate,
                            "structure_gate": structure_gate}
        return output


def parameter_count() -> int:
    return sum(value.numel() for value in OneStepActionTextureMemoryUNet().parameters())
