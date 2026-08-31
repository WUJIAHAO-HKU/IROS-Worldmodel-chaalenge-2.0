"""Action-pose-conditioned trajectory renderer for low/mid-frequency motion."""

from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


BETA_LEVELS = (.05, .10, .20, .40)
CLASS_COUNT = 1 + 5 * len(BETA_LEVELS)
OUTPUT_CHANNELS = 2 + 5 * len(BETA_LEVELS)


class TrajectoryMotionRenderer(nn.Module):
    patch_size = 8

    def __init__(self) -> None:
        super().__init__()
        # base RGB, five warped RGB, five disagreements = 33 visual;
        # actions14, horizon, arm and six projected pose coordinates = 22.
        self.network = nn.Sequential(
            nn.Conv3d(55, 96, 3, padding=1), nn.GroupNorm(12, 96), nn.GELU(),
            nn.Conv3d(96, 96, 3, padding=1), nn.GroupNorm(12, 96), nn.GELU(),
            nn.Conv3d(96, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv3d(64, OUTPUT_CHANNELS, 1),
        )
        nn.init.zeros_(self.network[-1].weight); nn.init.zeros_(self.network[-1].bias)
        with torch.no_grad(): self.network[-1].bias[0] = 2.0

    def forward(self, base: torch.Tensor, warped: torch.Tensor, actions: torch.Tensor,
                projected_pose: torch.Tensor) -> torch.Tensor:
        batch,time,_,height,width=base.shape
        if warped.shape!=(batch,time,5,3,height,width) or actions.shape!=(batch,time,14) or projected_pose.shape!=(batch,time,6):
            raise ValueError("trajectory motion input shape")
        visual=torch.cat((base,warped.flatten(2,3),(warped-base[:,:,None]).abs().flatten(2,3)),2)
        visual=F.avg_pool2d(visual.flatten(0,1),8,8).reshape(batch,time,33,height//8,width//8)
        horizon=torch.linspace(1/time,1,time,device=base.device,dtype=base.dtype)[None,:,None].expand(batch,-1,-1)
        delta=torch.cat((actions[:,:1],actions[:,1:]-actions[:,:-1]),1).abs();arm=(delta[:,:,7:].mean(2)>delta[:,:,:7].mean(2)).to(base.dtype)[:,:,None]
        condition=torch.cat((actions,horizon,arm,projected_pose),2)[:,:,:,None,None].expand(-1,-1,-1,*visual.shape[-2:])
        value=torch.cat((visual,condition),2).permute(0,2,1,3,4)
        return self.network(value).permute(0,2,1,3,4)


def decode_motion(logits:torch.Tensor)->torch.Tensor:
    active=logits[:,:,:2].argmax(2)>0;positive=logits[:,:,2:].argmax(2)+1
    return torch.where(active,positive,torch.zeros_like(positive))


def render_motion(base: torch.Tensor, warped: torch.Tensor, logits: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
    label=decode_motion(logits);active=label>0;index=(label-1).clamp_min(0);source=index//len(BETA_LEVELS);level=index%len(BETA_LEVELS)
    beta_values=torch.tensor(BETA_LEVELS,device=base.device,dtype=base.dtype);beta_patch=torch.where(active,beta_values[level],torch.zeros_like(base[:, :, 0, ::8, ::8]))
    source_patch=F.one_hot(source,5).permute(0,1,4,2,3).float();source_weight=F.interpolate(source_patch.flatten(0,1),size=base.shape[-2:],mode="nearest").reshape(*base.shape[:2],5,*base.shape[-2:])
    selected=(warped*source_weight[:,:,:,None]).sum(2);beta=F.interpolate(beta_patch.flatten(0,1)[:,None],size=base.shape[-2:],mode="nearest").reshape(*base.shape[:2],1,*base.shape[-2:])
    blur=lambda x:F.avg_pool2d(x,9,1,4,count_include_pad=False)
    residual=blur(selected.flatten(0,1))-blur(base.flatten(0,1));output=(base+beta*residual.reshape_as(base)).clamp(0,1)
    return output,beta


def parameter_count()->int:return sum(value.numel() for value in TrajectoryMotionRenderer().parameters())
