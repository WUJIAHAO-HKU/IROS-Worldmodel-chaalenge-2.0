"""Action-pose-conditioned residual flow correction for an eight-frame parent rollout."""

from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


class TrajectoryResidualFlow(nn.Module):
    def __init__(self,maximum_flow_pixels:float=8.) -> None:
        super().__init__();self.maximum_flow_pixels=maximum_flow_pixels
        self.network=nn.Sequential(
            nn.Conv3d(31,64,3,padding=1),nn.GroupNorm(8,64),nn.GELU(),
            nn.Conv3d(64,96,3,padding=1),nn.GroupNorm(12,96),nn.GELU(),
            nn.Conv3d(96,96,3,padding=1),nn.GroupNorm(12,96),nn.GELU(),
            nn.Conv3d(96,64,3,padding=1),nn.GroupNorm(8,64),nn.GELU(),
            nn.Conv3d(64,3,1),
        )
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
        with torch.no_grad():self.network[-1].bias[2]=-2.

    def forward(self,base:torch.Tensor,context_last:torch.Tensor,actions:torch.Tensor,projected_pose:torch.Tensor):
        batch,time,_,height,width=base.shape
        if context_last.shape!=(batch,3,height,width) or actions.shape!=(batch,time,14) or projected_pose.shape!=(batch,time,6):raise ValueError("residual flow input shape")
        previous=torch.cat((context_last[:,None],base[:,:-1]),1);visual=torch.cat((base,previous,base-previous),2)
        visual=F.interpolate(visual.flatten(0,1),size=(64,64),mode="bilinear",align_corners=False).reshape(batch,time,9,64,64)
        horizon=torch.linspace(1/time,1,time,device=base.device,dtype=base.dtype)[None,:,None].expand(batch,-1,-1);delta=torch.cat((actions[:,:1],actions[:,1:]-actions[:,:-1]),1).abs();arm=(delta[:,:,7:].mean(2)>delta[:,:,:7].mean(2)).to(base.dtype)[:,:,None]
        condition=torch.cat((actions,horizon,arm,projected_pose),2)[:,:,:,None,None].expand(-1,-1,-1,64,64);raw=self.network(torch.cat((visual,condition),2).permute(0,2,1,3,4));raw=F.interpolate(raw,size=(time,height,width),mode="trilinear",align_corners=False).permute(0,2,1,3,4)
        flow=self.maximum_flow_pixels*torch.tanh(raw[:,:,:2]);gate=raw[:,:,2:3].sigmoid();effective=flow*gate
        yy,xx=torch.meshgrid(torch.linspace(-1,1,height,device=base.device,dtype=base.dtype),torch.linspace(-1,1,width,device=base.device,dtype=base.dtype),indexing="ij");grid=torch.stack((xx,yy),-1)[None,None].expand(batch,time,-1,-1,-1).clone();grid[...,0]+=2*effective[:,:,0]/(width-1);grid[...,1]+=2*effective[:,:,1]/(height-1)
        flat=base.flatten(0,1);warped=F.grid_sample(flat,grid.flatten(0,1),mode="bilinear",padding_mode="border",align_corners=True).reshape_as(base)
        identity=F.grid_sample(flat,torch.stack((xx,yy),-1)[None].expand(batch*time,-1,-1,-1),mode="bilinear",padding_mode="border",align_corners=True).reshape_as(base)
        # Subtract grid_sample's identity interpolation round-off so zero flow
        # is bit-exactly the frozen parent while retaining flow gradients.
        output=(base+warped-identity).clamp(0,1)
        return output,{"flow":flow,"gate":gate,"effective_flow":effective}


def parameter_count()->int:return sum(p.numel() for p in TrajectoryResidualFlow().parameters())
