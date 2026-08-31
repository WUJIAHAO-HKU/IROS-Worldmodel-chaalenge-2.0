"""Online action-aware correction branch for a frozen autoregressive parent."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from .direct_latent_unet import Block

class OnlineParentCorrector(nn.Module):
 def __init__(self,base_channels=48,residual_scale=.12):
  super().__init__();c=base_channels;self.base_channels=c;self.residual_scale=residual_scale
  self.stem=nn.Conv2d(21,c,3,padding=1);self.e0=nn.Sequential(Block(c),Block(c));self.d1=nn.Conv2d(c,2*c,4,2,1);self.e1=nn.Sequential(Block(2*c),Block(2*c));self.d2=nn.Conv2d(2*c,4*c,4,2,1);self.e2=nn.Sequential(Block(4*c),Block(4*c));self.d3=nn.Conv2d(4*c,6*c,4,2,1);self.action=nn.Sequential(nn.Linear(5*14,6*c),nn.SiLU(),nn.Linear(6*c,6*c));self.mid=nn.Sequential(Block(6*c),Block(6*c),Block(6*c));self.u2=nn.Conv2d(10*c,4*c,3,padding=1);self.o2=nn.Sequential(Block(4*c),Block(4*c));self.u1=nn.Conv2d(6*c,2*c,3,padding=1);self.o1=nn.Sequential(Block(2*c),Block(2*c));self.u0=nn.Conv2d(3*c,c,3,padding=1);self.o0=nn.Sequential(Block(c),Block(c));self.output=nn.Conv2d(c,3,3,padding=1);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
 def forward(self,context,proposal,actions):
  motion=context[:,-1]-context[:,-2];x=torch.cat((context.flatten(1,2),proposal,proposal-context[:,-1],),1);x0=self.e0(self.stem(x));x1=self.e1(self.d1(x0));x2=self.e2(self.d2(x1));x3=self.d3(x2);condition=self.action(actions.flatten(1))[:,:,None,None];x=self.mid(x3+condition);x=F.interpolate(x,size=x2.shape[-2:],mode="bilinear",align_corners=False);x=self.o2(self.u2(torch.cat((x,x2),1)));x=F.interpolate(x,size=x1.shape[-2:],mode="bilinear",align_corners=False);x=self.o1(self.u1(torch.cat((x,x1),1)));x=F.interpolate(x,size=x0.shape[-2:],mode="bilinear",align_corners=False);x=self.o0(self.u0(torch.cat((x,x0),1)));correction=self.residual_scale*torch.tanh(self.output(x));return (proposal+correction).clamp(0,1),correction
