"""Semantic latent correction whose decoded delta is added to sharp RGB pixels."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from .direct_latent_unet import Block

class LatentSemanticDeltaRefiner(nn.Module):
 def __init__(self,base_channels=96):
  super().__init__();c=base_channels;self.base_channels=c
  self.stem=nn.Conv2d(52,c,3,padding=1);self.e0=nn.Sequential(Block(c),Block(c));self.d1=nn.Conv2d(c,2*c,4,2,1);self.e1=nn.Sequential(Block(2*c),Block(2*c));self.d2=nn.Conv2d(2*c,4*c,4,2,1);self.e2=nn.Sequential(Block(4*c),Block(4*c));self.action=nn.Sequential(nn.Linear(12*14,4*c),nn.SiLU(),nn.Linear(4*c,4*c));self.mid=nn.Sequential(Block(4*c),Block(4*c),Block(4*c));self.u1=nn.Conv2d(6*c,2*c,3,padding=1);self.o1=nn.Sequential(Block(2*c),Block(2*c));self.u0=nn.Conv2d(3*c,c,3,padding=1);self.o0=nn.Sequential(Block(c),Block(c));self.output=nn.Conv2d(c,32,3,padding=1);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
 def forward(self,context,baseline,history,future):
  b=baseline.shape[0];x=torch.cat((context.flatten(1,2),baseline.flatten(1,2)),1);x0=self.e0(self.stem(x));x1=self.e1(self.d1(x0));x2=self.e2(self.d2(x1));condition=self.action(torch.cat((history,future),1).flatten(1))[:,:,None,None];x=self.mid(x2+condition);x=F.interpolate(x,size=x1.shape[-2:],mode="bilinear",align_corners=False);x=self.o1(self.u1(torch.cat((x,x1),1)));x=F.interpolate(x,size=x0.shape[-2:],mode="bilinear",align_corners=False);x=self.o0(self.u0(torch.cat((x,x0),1)));delta=self.output(x).unflatten(1,(8,4));return baseline+delta,delta
