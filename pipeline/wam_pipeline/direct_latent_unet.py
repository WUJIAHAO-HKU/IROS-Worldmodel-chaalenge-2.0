"""High-capacity direct SDXL-latent world model for eight future frames."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(16, channels), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(16, channels), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1),
        )
    def forward(self, x): return x + self.net(x)


class DirectLatentUNet(nn.Module):
    context_frames=5; prediction_frames=8; latent_channels=4; action_dim=14
    def __init__(self, base_channels: int=128) -> None:
        super().__init__(); c=base_channels; self.base_channels=c
        self.stem=nn.Conv2d(20,c,3,padding=1); self.e0=nn.Sequential(Block(c),Block(c))
        self.d1=nn.Conv2d(c,2*c,4,2,1); self.e1=nn.Sequential(Block(2*c),Block(2*c))
        self.d2=nn.Conv2d(2*c,3*c,4,2,1); self.e2=nn.Sequential(Block(3*c),Block(3*c))
        self.d3=nn.Conv2d(3*c,4*c,4,2,1)
        self.action=nn.Sequential(nn.Linear(12*14,4*c),nn.SiLU(),nn.Linear(4*c,4*c))
        self.mid=nn.Sequential(Block(4*c),Block(4*c),Block(4*c),Block(4*c))
        self.u2=nn.Conv2d(7*c,3*c,3,padding=1); self.o2=nn.Sequential(Block(3*c),Block(3*c))
        self.u1=nn.Conv2d(5*c,2*c,3,padding=1); self.o1=nn.Sequential(Block(2*c),Block(2*c))
        self.u0=nn.Conv2d(3*c,c,3,padding=1); self.o0=nn.Sequential(Block(c),Block(c))
        self.output=nn.Conv2d(c,32,3,padding=1); nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
    def forward(self,context,history,future):
        b,_,_,h,w=context.shape; x0=self.e0(self.stem(context.flatten(1,2))); x1=self.e1(self.d1(x0)); x2=self.e2(self.d2(x1)); x3=self.d3(x2)
        condition=self.action(torch.cat((history,future),1).flatten(1))[:,:,None,None]; x=self.mid(x3+condition)
        x=F.interpolate(x,size=x2.shape[-2:],mode="bilinear",align_corners=False);x=self.o2(self.u2(torch.cat((x,x2),1)))
        x=F.interpolate(x,size=x1.shape[-2:],mode="bilinear",align_corners=False);x=self.o1(self.u1(torch.cat((x,x1),1)))
        x=F.interpolate(x,size=x0.shape[-2:],mode="bilinear",align_corners=False);x=self.o0(self.u0(torch.cat((x,x0),1)))
        residual=self.output(x).unflatten(1,(8,4));return context[:,-1:]+residual
