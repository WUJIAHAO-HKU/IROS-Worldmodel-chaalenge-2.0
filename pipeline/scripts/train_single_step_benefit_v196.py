#!/usr/bin/env python3
"""Stable single-step FP32 distillation for the persistent-memory benefit experts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Subset,WeightedRandomSampler

from wam_pipeline.autoregressive_texture_memory_v190 import OneStepActionTextureMemoryUNet,parameter_count
from train_autoregressive_texture_memory_v190 import evaluate,highpass,window_arm_labels
from train_autoregressive_unet import WindowDataset,frames_for_model,window_motion_scores


def continuous_gate_target(detail,truth,structure_scale):
    selected=detail["selected"].detach().float();base=detail["base"].detach().float();truth=truth.detach().float()
    texture_delta=highpass(selected)-highpass(base);structure_delta=structure_scale*(selected-base);residual=truth-base
    tt=texture_delta.square().sum(1)+1e-3;ss=structure_delta.square().sum(1)+1e-3;ts=(texture_delta*structure_delta).sum(1)
    tr=(texture_delta*residual).sum(1);sr=(structure_delta*residual).sum(1);det=(tt*ss-ts.square()).clamp_min(1e-6)
    texture=((tr*ss-sr*ts)/det).nan_to_num().clamp(0,1);structure=((sr*tt-tr*ts)/det).nan_to_num().clamp(0,1)
    texture=torch.where(texture>.02,texture,torch.zeros_like(texture));structure=torch.where(structure>.02,structure,torch.zeros_like(structure))
    return torch.stack((texture,structure),1)


def save(path,model,mean,std,metadata):
    path.mkdir(parents=True,exist_ok=True);temporary=path/"model.pt.tmp"
    model_format="track2-autoregressive-texture-memory-v19.9" if metadata.get("source_expert") else ("track2-autoregressive-texture-memory-v19.7" if metadata.get("direct_l1") else "track2-autoregressive-texture-memory-v19.6")
    torch.save({"format":model_format,"state_dict":model.state_dict()},temporary);os.replace(temporary,path/"model.pt")
    np.savez(path/"action_normalization.npz",mean=mean.cpu().numpy(),std=std.cpu().numpy())
    (path/"training_manifest.json").write_text(json.dumps(metadata,indent=2)+"\n")


def main():
    p=argparse.ArgumentParser();p.add_argument("--windows",required=True);p.add_argument("--split-manifest",required=True);p.add_argument("--init-checkpoint",required=True)
    p.add_argument("--memory-checkpoint",required=True);p.add_argument("--output",required=True);p.add_argument("--steps",type=int,default=500);p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--learning-rate",type=float,default=1e-4);p.add_argument("--validation-interval",type=int,default=100);p.add_argument("--validation-windows",type=int,default=16)
    p.add_argument("--direct-l1",action="store_true");p.add_argument("--decoupled-validation",action="store_true");p.add_argument("--use-source-expert",action="store_true")
    p.add_argument("--seed",type=int,default=20260808);p.add_argument("--device",default="cuda");args=p.parse_args();torch.manual_seed(args.seed);np.random.seed(args.seed)
    device=torch.device(args.device);split=json.loads(Path(args.split_manifest).read_text());train=WindowDataset(Path(args.windows),split["train_episodes"]);dev=WindowDataset(Path(args.windows),split["validation_episodes"])
    normalization=np.load(Path(args.init_checkpoint)/"action_normalization.npz");mean=torch.from_numpy(normalization["mean"]).to(device);std=torch.from_numpy(normalization["std"]).to(device)
    motion=window_motion_scores(train);arms=window_arm_labels(train);counts=np.bincount(arms,minlength=2);weights=(1+3*(motion>=.03))*(len(arms)/(2*counts[arms]))
    sampler=WeightedRandomSampler(torch.from_numpy(weights),len(train),replacement=True,generator=torch.Generator().manual_seed(args.seed));loader=DataLoader(train,batch_size=args.batch_size,sampler=sampler,num_workers=2,pin_memory=True)
    positions=np.linspace(0,len(dev)-1,min(len(dev),args.validation_windows),dtype=int).tolist();dev_loader=DataLoader(Subset(dev,positions),batch_size=1,num_workers=2)
    model=OneStepActionTextureMemoryUNet(use_source_expert=args.use_source_expert).to(device);parent=torch.load(Path(args.init_checkpoint)/"model.pt",map_location="cpu",weights_only=True);missing,unexpected=model.load_state_dict(parent["state_dict"],strict=False)
    if unexpected or any(not key.startswith(("memory_enc0","memory_fusion","memory_head","benefit_head","benefit_expert","source_expert")) for key in missing):raise ValueError((missing,unexpected))
    model.initialize_memory_encoder();memory_state=torch.load(Path(args.memory_checkpoint)/"model.pt",map_location="cpu",weights_only=True);missing,unexpected=model.load_state_dict(memory_state["state_dict"],strict=False)
    if unexpected or any(not key.startswith(("benefit_head","benefit_expert","source_expert")) for key in missing):raise ValueError((missing,unexpected))
    for name,value in model.named_parameters():value.requires_grad=name.startswith("benefit_expert")
    optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=args.learning_rate,weight_decay=1e-4);iterator=iter(loader);rng=np.random.default_rng(args.seed);history_log=[];best=-1e9;output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    for step in range(1,args.steps+1):
        try:context,history,future,target=next(iterator)
        except StopIteration:iterator=iter(loader);context,history,future,target=next(iterator)
        context=frames_for_model(context).to(device);memory=context.clone();target=frames_for_model(target).to(device);history=((history.to(device)-mean)/std).float();future=((future.to(device)-mean)/std).float();horizon=int(rng.integers(0,8))
        with torch.no_grad(),torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
            for index in range(horizon):
                _,detail=model(context,torch.cat((history,future[:,index:index+1]),1),memory,True);base=detail["base"].clamp(0,1)
                context=torch.cat((context[:,1:],base[:,None]),1);history=torch.cat((history[:,1:],future[:,index:index+1]),1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
            prediction,detail=model(context,torch.cat((history,future[:,horizon:horizon+1]),1),memory,True);prediction=prediction.clamp(0,1);truth=target[:,horizon]
        gate_target=continuous_gate_target(detail,truth,model.structure_scale);gate_logits=torch.cat((detail["texture_logit"],detail["structure_logit"]),1).float();gate=gate_logits.sigmoid()
        pixel=(prediction.float()-truth.float()).abs().mean();texture=(highpass(prediction.float())-highpass(truth.float())).abs().mean();gate_bce=F.binary_cross_entropy_with_logits(gate_logits,gate_target);gate_l1=F.smooth_l1_loss(gate,gate_target)
        accurate=(detail["base"].detach().float()-truth.float()).abs().mean(1,keepdim=True)<(2/255)
        protect=(prediction.float()-detail["base"].detach().float()).abs()[accurate.expand_as(prediction)].mean() if accurate.any() else pixel.new_zeros(())
        loss=(pixel+.5*texture+.1*protect+.001*gate.mean()+.005*gate_bce) if args.direct_l1 else (pixel+.1*gate_bce+.2*gate_l1)
        if not torch.isfinite(loss):raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward();gradient=float(torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],1));optimizer.step()
        if step==1 or step%20==0:print(json.dumps({"step":step,"horizon":horizon+1,"loss":float(loss),"pixel":float(pixel),"texture":float(texture),"protect":float(protect),"gate_bce":float(gate_bce),"gate_l1":float(gate_l1),"gate":float(gate.mean()),"target_gate":float(gate_target.mean()),"gradient":gradient,"peak_memory_gib":torch.cuda.max_memory_allocated()/2**30}),flush=True)
        if step%args.validation_interval==0 or step==args.steps:
            metrics=evaluate(dev_loader,model,device,mean,std,propagate_memory=not args.decoupled_validation);model.train();record={"step":step,"metrics":metrics};history_log.append(record);print(json.dumps(record),flush=True)
            score=min(min(v["improvement_percent"],v["texture_improvement_percent"]) for v in metrics["arms"].values());metadata={"format":"track2-v19.9-selector-plus-direct-gate-training" if args.use_source_expert else ("track2-v19.7-direct-single-step-training" if args.direct_l1 else "track2-v19.6-single-step-benefit-training"),"data_boundary":"supplied_50_episodes_only","step":step,"parameters":parameter_count(),"trainable_parameters":sum(v.numel() for v in model.parameters() if v.requires_grad),"train_windows":len(train),"dev_windows":len(dev),"single_step_fp32_gate":True,"direct_l1":args.direct_l1,"decoupled_validation":args.decoupled_validation,"source_expert":args.use_source_expert,"frozen_parent_alignment_selector":True,"history":history_log,"metrics":metrics}
            save(output/"latest",model,mean,std,metadata)
            if score>best:best=score;save(output/"best",model,mean,std,metadata)


if __name__=="__main__":main()
