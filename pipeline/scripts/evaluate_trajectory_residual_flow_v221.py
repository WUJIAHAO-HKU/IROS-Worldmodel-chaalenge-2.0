#!/usr/bin/env python3
"""Full FP32 evaluation of v22.1 flow alone and composed with v21.3 texture."""

from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from wam_pipeline.trajectory_residual_flow_v221 import TrajectoryResidualFlow
from wam_pipeline.trajectory_texture_selector_v210 import TrajectoryPatchSelector
from wam_pipeline.trajectory_benefit_renderer_v212 import TrajectoryBenefitRenderer,render
from train_autoregressive_texture_memory_v190 import highpass
from train_autoregressive_unet import WindowDataset,frames_for_model
from train_trajectory_benefit_renderer_v212 import select_sources
from train_trajectory_motion_renderer_v220 import load_pose,project_future
from calibrate_trajectory_renderer_v211 import load_parent,parent_trajectory_fp32,arm_index


REGION=(72,232,30,210)


def accumulator():return {"rgb":torch.zeros(8),"texture":torch.zeros(8),"temporal":torch.zeros(8),"contact":torch.zeros(8),"flow":0.,"arms":{0:[0.]*9,1:[0.]*9}}


def frame_metrics(value,target,context_last):
    y0,y1,x0,x1=REGION;rgb=(value-target).abs().mean((2,3,4)).cpu();texture=(highpass(value.flatten(0,1))-highpass(target.flatten(0,1))).abs().mean((1,2,3)).reshape(len(value),8).cpu();truth_delta=torch.cat((context_last[:,None],target),1).diff(dim=1);temporal=(torch.cat((context_last[:,None],value),1).diff(dim=1)-truth_delta).abs().mean((2,3,4)).cpu();contact=(value[:,:,:,y0:y1,x0:x1]-target[:,:,:,y0:y1,x0:x1]).abs().mean((2,3,4)).cpu();return rgb,texture,temporal,contact


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument("--windows",required=True);p.add_argument("--split-manifest",required=True);p.add_argument("--init-checkpoint",required=True);p.add_argument("--memory-checkpoint",required=True);p.add_argument("--pose-checkpoint",required=True);p.add_argument("--flow-checkpoints",nargs="+",required=True);p.add_argument("--selector-checkpoint",required=True);p.add_argument("--texture-checkpoint",required=True);p.add_argument("--output",required=True);args=p.parse_args();device=torch.device("cuda");split=json.loads(Path(args.split_manifest).read_text());data=WindowDataset(Path(args.windows),split["validation_episodes"]);loader=DataLoader(data,batch_size=1,num_workers=2,pin_memory=True);parent,mean,std=load_parent(Path(args.init_checkpoint),Path(args.memory_checkpoint),device);pose_model,pose_stats=load_pose(args.pose_checkpoint,device)
    selector=TrajectoryPatchSelector().to(device);selector.load_state_dict(torch.load(Path(args.selector_checkpoint)/"selector.pt",map_location="cpu",weights_only=True)["state_dict"]);selector.eval().requires_grad_(False);texture=TrajectoryBenefitRenderer().to(device);texture.load_state_dict(torch.load(Path(args.texture_checkpoint)/"renderer.pt",map_location="cpu",weights_only=True)["state_dict"]);texture.eval().requires_grad_(False);flows={}
    for path_value in args.flow_checkpoints:
        path=Path(path_value);model=TrajectoryResidualFlow().to(device);model.load_state_dict(torch.load(path/"flow_renderer.pt",map_location="cpu",weights_only=True)["state_dict"]);model.eval().requires_grad_(False);flows[f"{path.parent.parent.name}_{path.name}"]=model
    names=["texture_v213"]+[name for key in flows for name in (f"flow_{key}",f"combined_{key}")];values={name:accumulator() for name in names};parent_sum=accumulator();count=0
    for raw_context,raw_history,raw_future,raw_target in loader:
        arm=arm_index(raw_history,raw_future);raw_history=raw_history.to(device);raw_future=raw_future.to(device);context=frames_for_model(raw_context).to(device);target=frames_for_model(raw_target).to(device);pose,_=project_future(pose_model,pose_stats,raw_history,raw_future);history=((raw_history-mean)/std).float();future=((raw_future-mean)/std).float();base,warped=parent_trajectory_fp32(parent,context,history,future,context.clone());selected,probabilities=select_sources(selector,base,warped,future);texture_logits=texture(base.float(),selected.float(),probabilities.float(),future.float());texture_value,_=render(base.float(),selected.float(),texture_logits);outputs={"texture_v213":(texture_value,0.)}
        for key,model in flows.items():
            motion,detail=model(base.float(),context[:,-1].float(),future.float(),pose.float());outputs[f"flow_{key}"]=(motion,float(detail["effective_flow"].abs().mean()));outputs[f"combined_{key}"]=((motion+(texture_value-base)).clamp(0,1),float(detail["effective_flow"].abs().mean()))
        parent_metrics=frame_metrics(base,target,context[:,-1]);
        for field,item in zip(("rgb","texture","temporal","contact"),parent_metrics):parent_sum[field]+=255*item[0]
        for name,(output,flow_value) in outputs.items():
            metrics=frame_metrics(output,target,context[:,-1]);value=values[name]
            for field,item in zip(("rgb","texture","temporal","contact"),metrics):value[field]+=255*item[0]
            value["flow"]+=flow_value;record=value["arms"][arm]
            for index,(parent_item,item) in enumerate(zip(parent_metrics,metrics)):record[2*index]+=float(255*parent_item.mean());record[2*index+1]+=float(255*item.mean())
            record[8]+=1
        count+=1
        if count%50==0:print(json.dumps({"processed":count,"total":len(data)}),flush=True)
    results=[]
    for name,value in values.items():
        item={"name":name,"mean_effective_flow_px":value["flow"]/count,"arms":{}};scores=[]
        for field in ("rgb","texture","temporal","contact"):
            parent_metric=parent_sum[field]/count;metric=value[field]/count;improvement=100*(parent_metric-metric)/parent_metric;item[f"parent_{field}_mae"]=float(parent_metric.mean());item[f"{field}_mae"]=float(metric.mean());item[f"{field}_improvement_percent"]=float(100*(parent_metric.mean()-metric.mean())/parent_metric.mean());item[f"frame_{field}_improvement_percent"]=improvement.tolist();scores.extend((item[f"{field}_improvement_percent"],float(improvement.min())))
        for arm,record in value["arms"].items():
            n=record[8];result={"windows":int(n)}
            for index,field in enumerate(("rgb","texture","temporal","contact")):
                before=record[2*index]/n;after=record[2*index+1]/n;result[f"{field}_improvement_percent"]=100*(before-after)/before
            item["arms"][f"arm{arm}"]=result;scores.extend(v for k,v in result.items() if k.endswith("improvement_percent"))
        item["gate_score"]=min(scores);results.append(item)
    results.sort(key=lambda x:x["gate_score"],reverse=True);report={"format":"track2-v22.1-full-dev269-flow-and-v21.3-composition","data_boundary":"supplied_50_episodes_only","windows":len(data),"frames":len(data)*8,"results":results};out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(results),flush=True)


if __name__=="__main__":main()
