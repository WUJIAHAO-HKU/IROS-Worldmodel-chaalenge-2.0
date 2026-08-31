#!/usr/bin/env python3
"""Cache parent and v21.2 predictions for a continuous held-out episode."""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from wam_pipeline.trajectory_texture_selector_v210 import TrajectoryPatchSelector
from wam_pipeline.trajectory_benefit_renderer_v212 import TrajectoryBenefitRenderer, render
from train_autoregressive_unet import WindowDataset, frames_for_model
from train_trajectory_benefit_renderer_v212 import select_sources
from calibrate_trajectory_renderer_v211 import load_parent, parent_trajectory_fp32, arm_index


def rgb8(value: torch.Tensor) -> np.ndarray:
    return value.mul(255).round().clamp(0,255).byte().permute(0,1,3,4,2).cpu().numpy()


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument("--windows",required=True);p.add_argument("--episode",type=int,required=True);p.add_argument("--init-checkpoint",required=True);p.add_argument("--memory-checkpoint",required=True);p.add_argument("--selector-checkpoint",required=True);p.add_argument("--renderer-checkpoint",required=True);p.add_argument("--output-dir",required=True);args=p.parse_args();device=torch.device("cuda");data=WindowDataset(Path(args.windows),[args.episode]);loader=DataLoader(data,batch_size=1,num_workers=2,pin_memory=True);parent,mean,std=load_parent(Path(args.init_checkpoint),Path(args.memory_checkpoint),device);selector=TrajectoryPatchSelector().to(device);selector.load_state_dict(torch.load(Path(args.selector_checkpoint)/"selector.pt",map_location="cpu",weights_only=True)["state_dict"]);selector.eval().requires_grad_(False);renderer=TrajectoryBenefitRenderer().to(device);renderer.load_state_dict(torch.load(Path(args.renderer_checkpoint)/"renderer.pt",map_location="cpu",weights_only=True)["state_dict"]);renderer.eval().requires_grad_(False)
    parents=[];refined=[];targets=[];arms=[]
    for index,(raw_context,raw_history,raw_future,raw_target) in enumerate(loader):
        arms.append(arm_index(raw_history,raw_future));context=frames_for_model(raw_context).to(device);history=((raw_history.to(device)-mean)/std).float();future=((raw_future.to(device)-mean)/std).float();base,warped=parent_trajectory_fp32(parent,context,history,future,context.clone());selected,probabilities=select_sources(selector,base,warped,future);candidate,_=render(base.float(),selected.float(),renderer(base.float(),selected.float(),probabilities.float(),future.float()));parents.append(rgb8(base)[0]);refined.append(rgb8(candidate)[0]);targets.append(raw_target.numpy()[0])
        if (index+1)%50==0:print(json.dumps({"processed":index+1,"total":len(data)}),flush=True)
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True);names=np.asarray([path.name for path in data.paths]);parent_array=np.stack(parents);refined_array=np.stack(refined);target_array=np.stack(targets);np.savez_compressed(output/f"episode{args.episode}_parent.npz",prediction=parent_array,target=target_array,windows=names);np.savez_compressed(output/f"episode{args.episode}_v212.npz",prediction=refined_array,target=target_array,windows=names);summary={"format":"v21.2-continuous-cache","episode":args.episode,"windows":len(data),"arm0_windows":arms.count(0),"arm1_windows":arms.count(1),"parent_rgb_mae":float(np.abs(parent_array.astype(np.float32)-target_array).mean()),"v212_rgb_mae":float(np.abs(refined_array.astype(np.float32)-target_array).mean())};(output/f"episode{args.episode}_cache_metrics.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary),flush=True)


if __name__=="__main__":main()
