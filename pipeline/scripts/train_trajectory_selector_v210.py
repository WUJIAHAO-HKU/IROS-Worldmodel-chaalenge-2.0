#!/usr/bin/env python3
"""Train an eight-frame patch-coherent texture source selector."""

from __future__ import annotations
import argparse,json,os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Subset,WeightedRandomSampler

from wam_pipeline.autoregressive_texture_memory_v190 import OneStepActionTextureMemoryUNet
from wam_pipeline.trajectory_texture_selector_v210 import TrajectoryPatchSelector,temporally_smoothed_labels,parameter_count
from train_autoregressive_texture_memory_v190 import highpass,window_arm_labels
from train_autoregressive_unet import WindowDataset,frames_for_model,window_motion_scores


@torch.no_grad()
def parent_trajectory(model,context,history,future,memory):
    bases=[];warps=[]
    with torch.autocast("cuda",dtype=torch.bfloat16):
        for action in future.unbind(1):
            _,detail=model(context,torch.cat((history,action[:,None]),1),memory,True);base=detail["base"].clamp(0,1);bases.append(base);warps.append(detail["warped"].clamp(0,1));context=torch.cat((context[:,1:],base[:,None]),1);history=torch.cat((history[:,1:],action[:,None]),1)
    return torch.stack(bases,1),torch.stack(warps,1)


def patch_cost(warped,target):
    batch,time,sources,_,height,width=warped.shape;truth_hp=highpass(target.flatten(0,1)).reshape_as(target);warped_hp=highpass(warped.flatten(0,2)).reshape_as(warped)
    error=(warped-target[:,:,None]).abs().mean(3)+2*(warped_hp-truth_hp[:,:,None]).abs().mean(3)
    return F.avg_pool2d(error.flatten(0,2),8,8).reshape(batch,time,sources,height//8,width//8)


@torch.inference_mode()
def evaluate(loader,parent,selector,device,mean,std,switch_penalty):
    total_accuracy=0.;total_switch=0.;parent_error=0.;selected_error=0.;oracle_error=0.;count=0
    selector.eval()
    for context,history,future,target in loader:
        context=frames_for_model(context).to(device);memory=context.clone();target=frames_for_model(target).to(device);history=((history.to(device)-mean)/std).float();future=((future.to(device)-mean)/std).float();base,warped=parent_trajectory(parent,context,history,future,memory);cost=patch_cost(warped,target);labels=temporally_smoothed_labels(cost,switch_penalty);logits=selector(base.float(),warped.float(),future.float());prediction=logits.argmax(2)
        weights=F.interpolate(F.one_hot(prediction,5).permute(0,1,4,2,3).flatten(0,1).float(),size=base.shape[-2:],mode="nearest").reshape(len(base),8,5,*base.shape[-2:]);selected=(warped*weights[:,:,:,None]).sum(2)
        total_accuracy+=float((prediction==labels).float().mean())*len(base);total_switch+=float((prediction[:,1:]!=prediction[:,:-1]).float().mean())*len(base);parent_error+=float((base-target).abs().mean())*len(base);selected_error+=float((selected-target).abs().mean())*len(base);oracle_error+=float(cost.min(2).values.mean())*len(base);count+=len(base)
    return {"sample_count":count,"label_accuracy":total_accuracy/count,"switch_rate":total_switch/count,"parent_rgb_mae":255*parent_error/count,"selected_rgb_mae":255*selected_error/count,"patch_oracle_combined_error":oracle_error/count}


def save(path,selector,metadata):
    path.mkdir(parents=True,exist_ok=True);temporary=path/"selector.pt.tmp";torch.save({"format":"track2-trajectory-texture-selector-v21.0","state_dict":selector.state_dict()},temporary);os.replace(temporary,path/"selector.pt");(path/"training_manifest.json").write_text(json.dumps(metadata,indent=2)+"\n")


def main():
    p=argparse.ArgumentParser();p.add_argument("--windows",required=True);p.add_argument("--split-manifest",required=True);p.add_argument("--init-checkpoint",required=True);p.add_argument("--memory-checkpoint",required=True);p.add_argument("--output",required=True);p.add_argument("--steps",type=int,default=300);p.add_argument("--batch-size",type=int,default=4);p.add_argument("--learning-rate",type=float,default=2e-4);p.add_argument("--validation-interval",type=int,default=50);p.add_argument("--validation-windows",type=int,default=16);p.add_argument("--switch-penalty",type=float,default=.01);p.add_argument("--seed",type=int,default=20260808);args=p.parse_args();torch.manual_seed(args.seed);np.random.seed(args.seed);device=torch.device("cuda")
    split=json.loads(Path(args.split_manifest).read_text());train=WindowDataset(Path(args.windows),split["train_episodes"]);dev=WindowDataset(Path(args.windows),split["validation_episodes"]);norm=np.load(Path(args.init_checkpoint)/"action_normalization.npz");mean=torch.from_numpy(norm["mean"]).to(device);std=torch.from_numpy(norm["std"]).to(device);motion=window_motion_scores(train);arms=window_arm_labels(train);counts=np.bincount(arms,minlength=2);weights=(1+3*(motion>=.03))*(len(arms)/(2*counts[arms]));sampler=WeightedRandomSampler(torch.from_numpy(weights),len(train),replacement=True,generator=torch.Generator().manual_seed(args.seed));loader=DataLoader(train,batch_size=args.batch_size,sampler=sampler,num_workers=2,pin_memory=True);iterator=iter(loader);positions=np.linspace(0,len(dev)-1,min(len(dev),args.validation_windows),dtype=int).tolist();dev_loader=DataLoader(Subset(dev,positions),batch_size=1,num_workers=2)
    parent=OneStepActionTextureMemoryUNet().to(device);state=torch.load(Path(args.init_checkpoint)/"model.pt",map_location="cpu",weights_only=True);missing,unexpected=parent.load_state_dict(state["state_dict"],strict=False);allowed=("memory_enc0","memory_fusion","memory_head","benefit_head","benefit_expert","source_expert")
    if unexpected or any(not key.startswith(allowed) for key in missing):raise ValueError((missing,unexpected))
    parent.initialize_memory_encoder();memory=torch.load(Path(args.memory_checkpoint)/"model.pt",map_location="cpu",weights_only=True);missing,unexpected=parent.load_state_dict(memory["state_dict"],strict=False)
    if unexpected or any(not key.startswith(("benefit_head","benefit_expert","source_expert")) for key in missing):raise ValueError((missing,unexpected))
    parent.eval();parent.requires_grad_(False);selector=TrajectoryPatchSelector().to(device);optimizer=torch.optim.AdamW(selector.parameters(),lr=args.learning_rate,weight_decay=1e-4);history_log=[];best=1e9;output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    for step in range(1,args.steps+1):
        try:context,history,future,target=next(iterator)
        except StopIteration:iterator=iter(loader);context,history,future,target=next(iterator)
        context=frames_for_model(context).to(device);memory_frames=context.clone();target=frames_for_model(target).to(device);history=((history.to(device)-mean)/std).float();future=((future.to(device)-mean)/std).float();base,warped=parent_trajectory(parent,context,history,future,memory_frames);cost=patch_cost(warped,target).detach();labels=temporally_smoothed_labels(cost,args.switch_penalty);optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):logits=selector(base.float(),warped.float(),future.float());loss=F.cross_entropy(logits.permute(0,2,1,3,4).float(),labels)
        if not torch.isfinite(loss):raise FloatingPointError(step)
        loss.backward();gradient=float(torch.nn.utils.clip_grad_norm_(selector.parameters(),1));optimizer.step();accuracy=float((logits.argmax(2)==labels).float().mean());switch=float((logits.argmax(2)[:,1:]!=logits.argmax(2)[:,:-1]).float().mean())
        if step==1 or step%20==0:print(json.dumps({"step":step,"loss":float(loss),"accuracy":accuracy,"switch_rate":switch,"label_switch_rate":float((labels[:,1:]!=labels[:,:-1]).float().mean()),"gradient":gradient,"peak_memory_gib":torch.cuda.max_memory_allocated()/2**30}),flush=True)
        if step%args.validation_interval==0 or step==args.steps:
            metrics=evaluate(dev_loader,parent,selector,device,mean,std,args.switch_penalty);selector.train();record={"step":step,"metrics":metrics};history_log.append(record);print(json.dumps(record),flush=True);metadata={"format":"track2-v21.0-trajectory-selector-training","data_boundary":"supplied_50_episodes_only","step":step,"parameters":parameter_count(),"train_windows":len(train),"dev_windows":len(dev),"patch_size":8,"switch_penalty":args.switch_penalty,"history":history_log,"metrics":metrics};save(output/"latest",selector,metadata)
            if metrics["selected_rgb_mae"]<best:best=metrics["selected_rgb_mae"];save(output/"best",selector,metadata)


if __name__=="__main__":main()
