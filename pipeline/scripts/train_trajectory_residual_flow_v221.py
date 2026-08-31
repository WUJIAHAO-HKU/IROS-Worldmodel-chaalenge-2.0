#!/usr/bin/env python3
"""Train v22.1 residual flow using frozen parent rollouts and action pose."""

from __future__ import annotations
import argparse,json,os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Subset,WeightedRandomSampler

from wam_pipeline.autoregressive_texture_memory_v190 import OneStepActionTextureMemoryUNet
from wam_pipeline.trajectory_residual_flow_v221 import TrajectoryResidualFlow,parameter_count
from train_autoregressive_texture_memory_v190 import highpass,window_arm_labels
from train_autoregressive_unet import WindowDataset,frames_for_model,window_motion_scores
from train_trajectory_selector_v210 import parent_trajectory
from train_trajectory_motion_renderer_v220 import load_pose,project_future


def errors(value,target,context_last):
    rgb=(value-target).abs().mean();texture=(highpass(value.flatten(0,1))-highpass(target.flatten(0,1))).abs().mean();truth_delta=torch.cat((context_last[:,None],target),1).diff(dim=1);delta=(torch.cat((context_last[:,None],value),1).diff(dim=1)-truth_delta).abs().mean();return rgb,texture,delta


def objective(output,base,target,context_last,detail):
    previous_target=torch.cat((context_last[:,None],target[:,:-1]),1);motion=(target-previous_target).abs().mean(2,keepdim=True);weight=1+1.5*F.max_pool2d((motion>.012).flatten(0,1).float(),7,1,3).reshape_as(motion);pixel=((output-target).abs()*weight).sum()/(weight.sum()*3);high=(highpass(output.flatten(0,1))-highpass(target.flatten(0,1))).abs().mean();truth_delta=torch.cat((context_last[:,None],target),1).diff(dim=1);temporal=(torch.cat((context_last[:,None],output),1).diff(dim=1)-truth_delta).abs().mean();accurate=(base-target).abs().mean(2,keepdim=True)<(2/255);protect=(output-base).abs()[accurate.expand_as(output)].mean() if accurate.any() else output.new_zeros(());flow=detail["effective_flow"];smooth=(flow[:,:,:,1:]-flow[:,:,:,:-1]).abs().mean()+(flow[:,:,:,:,1:]-flow[:,:,:,:,:-1]).abs().mean();magnitude=flow.abs().mean();loss=pixel+.3*temporal+.1*high+.4*protect+.01*smooth+.001*magnitude;return loss,{"pixel":pixel,"temporal":temporal,"high":high,"protect":protect,"smooth":smooth,"flow":magnitude}


@torch.inference_mode()
def evaluate(loader,parent,pose_model,pose_stats,model,device,mean,std):
    model.eval();sums=np.zeros(9,np.float64);count=0
    for context,history,future,target in loader:
        raw_history=history.to(device);raw_future=future.to(device);context=frames_for_model(context).to(device);target=frames_for_model(target).to(device);pose,_=project_future(pose_model,pose_stats,raw_history,raw_future);history=((raw_history-mean)/std).float();future=((raw_future-mean)/std).float();base,_=parent_trajectory(parent,context,history,future,context.clone());output,detail=model(base.float(),context[:,-1].float(),future.float(),pose.float());parent_values=errors(base,target,context[:,-1]);values=errors(output,target,context[:,-1]);sums[:3]+=np.asarray([float(x) for x in parent_values]);sums[3:6]+=np.asarray([float(x) for x in values]);sums[6]+=float(detail["effective_flow"].abs().mean());sums[7]+=float(detail["gate"].mean());sums[8]+=float((detail["effective_flow"].square().sum(2).sqrt()>.25).float().mean());count+=1
    sums/=count;return {"windows":count,"parent_rgb_mae":255*sums[0],"rgb_mae":255*sums[3],"rgb_improvement_percent":100*(sums[0]-sums[3])/sums[0],"parent_texture_mae":255*sums[1],"texture_mae":255*sums[4],"texture_improvement_percent":100*(sums[1]-sums[4])/sums[1],"parent_temporal_delta_mae":255*sums[2],"temporal_delta_mae":255*sums[5],"temporal_improvement_percent":100*(sums[2]-sums[5])/sums[2],"mean_effective_flow_px":sums[6],"mean_gate":sums[7],"moving_pixel_rate":sums[8]}


def save(path,model,metadata):
    path.mkdir(parents=True,exist_ok=True);tmp=path/"flow_renderer.pt.tmp";torch.save({"format":"track2-trajectory-residual-flow-v22.1","state_dict":model.state_dict()},tmp);os.replace(tmp,path/"flow_renderer.pt");(path/"training_manifest.json").write_text(json.dumps(metadata,indent=2)+"\n")


def main():
    p=argparse.ArgumentParser();p.add_argument("--windows",required=True);p.add_argument("--split-manifest",required=True);p.add_argument("--init-checkpoint",required=True);p.add_argument("--memory-checkpoint",required=True);p.add_argument("--pose-checkpoint",required=True);p.add_argument("--output",required=True);p.add_argument("--steps",type=int,default=400);p.add_argument("--batch-size",type=int,default=16);p.add_argument("--learning-rate",type=float,default=2e-4);p.add_argument("--validation-interval",type=int,default=50);p.add_argument("--validation-windows",type=int,default=16);p.add_argument("--train-arm",type=int,choices=(0,1));p.add_argument("--seed",type=int,default=20260808);args=p.parse_args();torch.manual_seed(args.seed);np.random.seed(args.seed);device=torch.device("cuda");split=json.loads(Path(args.split_manifest).read_text());train=WindowDataset(Path(args.windows),split["train_episodes"]);dev=WindowDataset(Path(args.windows),split["validation_episodes"]);norm=np.load(Path(args.init_checkpoint)/"action_normalization.npz");mean=torch.from_numpy(norm["mean"]).to(device);std=torch.from_numpy(norm["std"]).to(device);motion=window_motion_scores(train);arms=window_arm_labels(train);counts=np.bincount(arms,minlength=2);weights=(1+3*(motion>=.03))*(len(arms)/(2*counts[arms]));weights=weights if args.train_arm is None else weights*(arms==args.train_arm);sampler=WeightedRandomSampler(torch.from_numpy(weights),len(train),replacement=True,generator=torch.Generator().manual_seed(args.seed));loader=DataLoader(train,batch_size=args.batch_size,sampler=sampler,num_workers=2,pin_memory=True);iterator=iter(loader);positions=np.linspace(0,len(dev)-1,min(len(dev),args.validation_windows),dtype=int).tolist();dev_loader=DataLoader(Subset(dev,positions),batch_size=1,num_workers=2)
    parent=OneStepActionTextureMemoryUNet().to(device);state=torch.load(Path(args.init_checkpoint)/"model.pt",map_location="cpu",weights_only=True);parent.load_state_dict(state["state_dict"],strict=False);parent.initialize_memory_encoder();memory=torch.load(Path(args.memory_checkpoint)/"model.pt",map_location="cpu",weights_only=True);parent.load_state_dict(memory["state_dict"],strict=False);parent.eval().requires_grad_(False);pose_model,pose_stats=load_pose(args.pose_checkpoint,device);model=TrajectoryResidualFlow().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-4);output=Path(args.output);history_log=[];best=-1e9
    for step in range(1,args.steps+1):
        try:context,history,future,target=next(iterator)
        except StopIteration:iterator=iter(loader);context,history,future,target=next(iterator)
        raw_history=history.to(device);raw_future=future.to(device);context=frames_for_model(context).to(device);target=frames_for_model(target).to(device);pose,_=project_future(pose_model,pose_stats,raw_history,raw_future);history=((raw_history-mean)/std).float();future=((raw_future-mean)/std).float();base,_=parent_trajectory(parent,context,history,future,context.clone());optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):output_frame,detail=model(base.float(),context[:,-1].float(),future.float(),pose.float());loss,parts=objective(output_frame,target=target,base=base,context_last=context[:,-1],detail=detail)
        loss.backward();gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1));optimizer.step()
        if step==1 or step%20==0:print(json.dumps({"step":step,"loss":float(loss),**{k:float(v) for k,v in parts.items()},"gradient":gradient,"peak_memory_gib":torch.cuda.max_memory_allocated()/2**30}),flush=True)
        if step%args.validation_interval==0 or step==args.steps:
            metrics=evaluate(dev_loader,parent,pose_model,pose_stats,model,device,mean,std);model.train();record={"step":step,"metrics":metrics};history_log.append(record);print(json.dumps(record),flush=True);metadata={"format":"track2-v22.1-residual-flow-training","data_boundary":"supplied_50_episodes_only","step":step,"parameters":parameter_count(),"train_arm":args.train_arm,"train_windows":len(train),"dev_windows":len(dev),"history":history_log,"metrics":metrics};save(output/f"step_{step:04d}",model,metadata);save(output/"latest",model,metadata);score=min(metrics["rgb_improvement_percent"],metrics["temporal_improvement_percent"])
            if score>best:best=score;save(output/"best",model,metadata)


if __name__=="__main__":main()
