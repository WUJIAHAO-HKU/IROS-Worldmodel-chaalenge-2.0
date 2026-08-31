#!/usr/bin/env python3
"""Cache continuous parent and v22.2 residual-flow predictions."""

from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from wam_pipeline.trajectory_residual_flow_v221 import TrajectoryResidualFlow
from train_autoregressive_unet import WindowDataset,frames_for_model
from train_trajectory_motion_renderer_v220 import load_pose,project_future
from calibrate_trajectory_renderer_v211 import load_parent,parent_trajectory_fp32,arm_index


def rgb8(value):return value.mul(255).round().clamp(0,255).byte().permute(0,1,3,4,2).cpu().numpy()


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument("--windows",required=True);p.add_argument("--episode",type=int,required=True);p.add_argument("--init-checkpoint",required=True);p.add_argument("--memory-checkpoint",required=True);p.add_argument("--pose-checkpoint",required=True);p.add_argument("--flow-checkpoint",required=True);p.add_argument("--output-dir",required=True);args=p.parse_args();device=torch.device("cuda");data=WindowDataset(Path(args.windows),[args.episode]);loader=DataLoader(data,batch_size=1,num_workers=2,pin_memory=True);parent,mean,std=load_parent(Path(args.init_checkpoint),Path(args.memory_checkpoint),device);pose_model,pose_stats=load_pose(args.pose_checkpoint,device);model=TrajectoryResidualFlow().to(device);model.load_state_dict(torch.load(Path(args.flow_checkpoint)/"flow_renderer.pt",map_location="cpu",weights_only=True)["state_dict"]);model.eval().requires_grad_(False);parents=[];predictions=[];targets=[];arms=[]
    for index,(raw_context,raw_history,raw_future,raw_target) in enumerate(loader):
        arms.append(arm_index(raw_history,raw_future));raw_history=raw_history.to(device);raw_future=raw_future.to(device);context=frames_for_model(raw_context).to(device);pose,_=project_future(pose_model,pose_stats,raw_history,raw_future);history=((raw_history-mean)/std).float();future=((raw_future-mean)/std).float();base,_=parent_trajectory_fp32(parent,context,history,future,context.clone());value,_=model(base.float(),context[:,-1].float(),future.float(),pose.float());parents.append(rgb8(base)[0]);predictions.append(rgb8(value)[0]);targets.append(raw_target.numpy()[0]);
        if (index+1)%50==0:print(json.dumps({"processed":index+1,"total":len(data)}),flush=True)
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True);names=np.asarray([path.name for path in data.paths]);parent_array=np.stack(parents);prediction_array=np.stack(predictions);target_array=np.stack(targets);np.savez_compressed(output/f"episode{args.episode}_parent.npz",prediction=parent_array,target=target_array,windows=names);np.savez_compressed(output/f"episode{args.episode}_v222.npz",prediction=prediction_array,target=target_array,windows=names);result={"format":"v22.2-continuous-cache","episode":args.episode,"windows":len(data),"arm0_windows":arms.count(0),"arm1_windows":arms.count(1),"parent_rgb_mae":float(np.abs(parent_array.astype(np.float32)-target_array).mean()),"v222_rgb_mae":float(np.abs(prediction_array.astype(np.float32)-target_array).mean())};(output/f"episode{args.episode}_cache_metrics.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result),flush=True)


if __name__=="__main__":main()
