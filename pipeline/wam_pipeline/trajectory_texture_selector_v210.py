"""Patch-coherent eight-frame source selector for persistent texture memory."""

from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


class TrajectoryPatchSelector(nn.Module):
    patch_size=8

    def __init__(self) -> None:
        super().__init__()
        # base RGB + five warped RGB + five RGB disagreements + action + horizon + arm.
        self.network=nn.Sequential(
            nn.Conv3d(49,64,3,padding=1),nn.GroupNorm(8,64),nn.GELU(),
            nn.Conv3d(64,64,3,padding=1),nn.GroupNorm(8,64),nn.GELU(),
            nn.Conv3d(64,48,3,padding=1),nn.GroupNorm(8,48),nn.GELU(),
            nn.Conv3d(48,5,1),
        )
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)

    def forward(self,base:torch.Tensor,warped:torch.Tensor,actions:torch.Tensor)->torch.Tensor:
        batch,time,_,height,width=base.shape
        if warped.shape!=(batch,time,5,3,height,width) or actions.shape!=(batch,time,14):raise ValueError("trajectory input shape")
        visual=torch.cat((base,warped.flatten(2,3),(warped-base[:,:,None]).abs().flatten(2,3)),2)
        visual=F.avg_pool2d(visual.flatten(0,1),self.patch_size,self.patch_size).reshape(batch,time,33,height//self.patch_size,width//self.patch_size)
        horizon=torch.linspace(1/8,1,time,device=base.device,dtype=base.dtype)[None,:,None].expand(batch,-1,-1)
        delta=torch.cat((actions[:,:1],actions[:,1:]-actions[:,:-1]),1).abs();arm=(delta[:,:,7:].mean(2)>delta[:,:,:7].mean(2)).to(base.dtype)[:,:,None]
        condition=torch.cat((actions,horizon,arm),2)[:,:,:,None,None].expand(-1,-1,-1,*visual.shape[-2:])
        return self.network(torch.cat((visual,condition),2).permute(0,2,1,3,4)).permute(0,2,1,3,4)


def temporally_smoothed_labels(cost:torch.Tensor,switch_penalty:float=.01)->torch.Tensor:
    """Viterbi labels for [B,T,5,H,W] costs."""
    dp=cost[:,0];back=[]
    for index in range(1,cost.shape[1]):
        minimum,minimum_source=dp.min(1,keepdim=True);switch=minimum+switch_penalty
        stay_better=dp<=switch;predecessor=torch.where(stay_better,torch.arange(5,device=cost.device)[None,:,None,None],minimum_source.expand_as(dp))
        dp=cost[:,index]+torch.where(stay_better,dp,switch);back.append(predecessor)
    state=dp.argmin(1);labels=[state]
    for predecessor in reversed(back):
        state=predecessor.gather(1,state[:,None]).squeeze(1);labels.append(state)
    return torch.stack(list(reversed(labels)),1)


def parameter_count()->int:return sum(value.numel() for value in TrajectoryPatchSelector().parameters())
