#!/usr/bin/env python3
"""Compare every v21.2 renderer checkpoint on the complete FP32 dev set."""

from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from wam_pipeline.trajectory_texture_selector_v210 import TrajectoryPatchSelector
from wam_pipeline.trajectory_benefit_renderer_v212 import TrajectoryBenefitRenderer, render
from train_autoregressive_texture_memory_v190 import highpass
from train_autoregressive_unet import WindowDataset, frames_for_model
from train_trajectory_benefit_renderer_v212 import select_sources
from calibrate_trajectory_renderer_v211 import load_parent, parent_trajectory_fp32, arm_index


def accumulator():
    return {"rgb": torch.zeros(8), "texture": torch.zeros(8), "alpha": torch.zeros(8),
            "arms": {0: [0.,0.,0.,0.,0], 1: [0.,0.,0.,0.,0.]}}


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument("--windows",required=True);p.add_argument("--split-manifest",required=True);p.add_argument("--init-checkpoint",required=True);p.add_argument("--memory-checkpoint",required=True);p.add_argument("--selector-checkpoint",required=True);p.add_argument("--renderer-root",required=True);p.add_argument("--output",required=True);args=p.parse_args();device=torch.device("cuda")
    split=json.loads(Path(args.split_manifest).read_text());data=WindowDataset(Path(args.windows),split["validation_episodes"]);loader=DataLoader(data,batch_size=1,num_workers=2,pin_memory=True);parent,mean,std=load_parent(Path(args.init_checkpoint),Path(args.memory_checkpoint),device)
    selector=TrajectoryPatchSelector().to(device);selector.load_state_dict(torch.load(Path(args.selector_checkpoint)/"selector.pt",map_location="cpu",weights_only=True)["state_dict"]);selector.eval().requires_grad_(False)
    paths=sorted(path for path in Path(args.renderer_root).glob("step_*") if int(path.name.split("_")[1])>=100);renderers={}
    for path in paths:
        model=TrajectoryBenefitRenderer().to(device);model.load_state_dict(torch.load(path/"renderer.pt",map_location="cpu",weights_only=True)["state_dict"]);model.eval().requires_grad_(False);renderers[path.name]=model
    values={name:accumulator() for name in renderers};parent_rgb=torch.zeros(8);parent_texture=torch.zeros(8);count=0
    for raw_context,raw_history,raw_future,raw_target in loader:
        arm=arm_index(raw_history,raw_future);context=frames_for_model(raw_context).to(device);target=frames_for_model(raw_target).to(device);history=((raw_history.to(device)-mean)/std).float();future=((raw_future.to(device)-mean)/std).float();base,warped=parent_trajectory_fp32(parent,context,history,future,context.clone());selected,probabilities=select_sources(selector,base,warped,future);pe=(base-target).abs().mean((2,3,4)).cpu();target_hp=highpass(target.flatten(0,1)).reshape_as(target);pte=(highpass(base.flatten(0,1)).reshape_as(base)-target_hp).abs().mean((2,3,4)).cpu();parent_rgb+=255*pe[0];parent_texture+=255*pte[0]
        for name,model in renderers.items():
            logits=model(base.float(),selected.float(),probabilities.float(),future.float());candidate,alpha=render(base.float(),selected.float(),logits);rgb=(candidate-target).abs().mean((2,3,4)).cpu();texture=(highpass(candidate.flatten(0,1)).reshape_as(candidate)-target_hp).abs().mean((2,3,4)).cpu();value=values[name];value["rgb"]+=255*rgb[0];value["texture"]+=255*texture[0];value["alpha"]+=alpha.mean((0,2,3,4)).cpu();record=value["arms"][arm];record[0]+=float(255*pe.mean());record[1]+=float(255*rgb.mean());record[2]+=float(255*pte.mean());record[3]+=float(255*texture.mean());record[4]+=1
        count+=1
        if count%50==0:print(json.dumps({"processed":count,"total":len(data)}),flush=True)
    parent_rgb/=count;parent_texture/=count;results=[]
    for name,value in values.items():
        rgb=value["rgb"]/count;texture=value["texture"]/count;frame=100*(parent_rgb-rgb)/parent_rgb;item={"checkpoint":name,"rgb_mae":float(rgb.mean()),"texture_mae":float(texture.mean()),"rgb_improvement_percent":float(100*(parent_rgb.mean()-rgb.mean())/parent_rgb.mean()),"texture_improvement_percent":float(100*(parent_texture.mean()-texture.mean())/parent_texture.mean()),"frame_rgb_mae":rgb.tolist(),"frame_rgb_improvement_percent":frame.tolist(),"frame_mean_alpha":(value["alpha"]/count).tolist(),"arms":{}};scores=[item["rgb_improvement_percent"],item["texture_improvement_percent"],float(frame[3:].min())]
        for arm,(pr,vr,pt,vt,n) in value["arms"].items():
            result={"windows":int(n),"parent_rgb_mae":pr/n,"rgb_mae":vr/n,"parent_texture_mae":pt/n,"texture_mae":vt/n,"rgb_improvement_percent":100*(pr-vr)/pr,"texture_improvement_percent":100*(pt-vt)/pt};item["arms"][f"arm{arm}"]=result;scores.extend((result["rgb_improvement_percent"],result["texture_improvement_percent"]))
        item["gate_score"]=min(scores);results.append(item)
    results.sort(key=lambda x:x["gate_score"],reverse=True);report={"format":"v21.2-trajectory-benefit-renderer-full-dev269","data_boundary":"supplied_50_episodes_only","episodes":split["validation_episodes"],"windows":len(data),"frames":len(data)*8,"parent_rgb_mae":float(parent_rgb.mean()),"parent_texture_mae":float(parent_texture.mean()),"parent_frame_rgb_mae":parent_rgb.tolist(),"results":results};output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(results),flush=True)


if __name__=="__main__":main()
