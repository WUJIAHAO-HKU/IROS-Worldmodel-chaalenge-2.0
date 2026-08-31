#!/usr/bin/env python3
"""Train a target-supervised five-source selector without recursive gradients."""

from __future__ import annotations

import argparse,json,os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Subset,WeightedRandomSampler

from wam_pipeline.autoregressive_texture_memory_v190 import OneStepActionTextureMemoryUNet,parameter_count
from train_autoregressive_texture_memory_v190 import evaluate,highpass,window_arm_labels
from train_autoregressive_unet import WindowDataset,frames_for_model,window_motion_scores


def save(path,model,mean,std,metadata):
    path.mkdir(parents=True,exist_ok=True);temporary=path/"model.pt.tmp";torch.save({"format":"track2-autoregressive-texture-memory-v19.8","state_dict":model.state_dict()},temporary);os.replace(temporary,path/"model.pt")
    np.savez(path/"action_normalization.npz",mean=mean.cpu().numpy(),std=std.cpu().numpy());(path/"training_manifest.json").write_text(json.dumps(metadata,indent=2)+"\n")


def main():
    p=argparse.ArgumentParser();p.add_argument("--windows",required=True);p.add_argument("--split-manifest",required=True);p.add_argument("--init-checkpoint",required=True);p.add_argument("--memory-checkpoint",required=True);p.add_argument("--output",required=True)
    p.add_argument("--steps",type=int,default=400);p.add_argument("--batch-size",type=int,default=32);p.add_argument("--learning-rate",type=float,default=1e-4);p.add_argument("--validation-interval",type=int,default=50);p.add_argument("--validation-windows",type=int,default=16);p.add_argument("--seed",type=int,default=20260808);args=p.parse_args()
    torch.manual_seed(args.seed);np.random.seed(args.seed);device=torch.device("cuda");split=json.loads(Path(args.split_manifest).read_text());train=WindowDataset(Path(args.windows),split["train_episodes"]);dev=WindowDataset(Path(args.windows),split["validation_episodes"])
    norm=np.load(Path(args.init_checkpoint)/"action_normalization.npz");mean=torch.from_numpy(norm["mean"]).to(device);std=torch.from_numpy(norm["std"]).to(device);motion=window_motion_scores(train);arms=window_arm_labels(train);counts=np.bincount(arms,minlength=2);weights=(1+3*(motion>=.03))*(len(arms)/(2*counts[arms]))
    sampler=WeightedRandomSampler(torch.from_numpy(weights),len(train),replacement=True,generator=torch.Generator().manual_seed(args.seed));loader=DataLoader(train,batch_size=args.batch_size,sampler=sampler,num_workers=2,pin_memory=True);iterator=iter(loader)
    positions=np.linspace(0,len(dev)-1,min(len(dev),args.validation_windows),dtype=int).tolist();dev_loader=DataLoader(Subset(dev,positions),batch_size=1,num_workers=2)
    model=OneStepActionTextureMemoryUNet(use_source_expert=True).to(device);parent=torch.load(Path(args.init_checkpoint)/"model.pt",map_location="cpu",weights_only=True);missing,unexpected=model.load_state_dict(parent["state_dict"],strict=False)
    allowed=("memory_enc0","memory_fusion","memory_head","benefit_head","benefit_expert","source_expert")
    if unexpected or any(not key.startswith(allowed) for key in missing):raise ValueError((missing,unexpected))
    model.initialize_memory_encoder()
    memory_state=torch.load(Path(args.memory_checkpoint)/"model.pt",map_location="cpu",weights_only=True);missing,unexpected=model.load_state_dict(memory_state["state_dict"],strict=False)
    if unexpected or any(not key.startswith(("benefit_head","benefit_expert","source_expert")) for key in missing):raise ValueError((missing,unexpected))
    for name,value in model.named_parameters():value.requires_grad=name.startswith("source_expert")
    optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=args.learning_rate,weight_decay=1e-4);rng=np.random.default_rng(args.seed);history_log=[];best=-1e9;output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    for step in range(1,args.steps+1):
        try:context,history,future,target=next(iterator)
        except StopIteration:iterator=iter(loader);context,history,future,target=next(iterator)
        context=frames_for_model(context).to(device);memory=context.clone();target=frames_for_model(target).to(device);history=((history.to(device)-mean)/std).float();future=((future.to(device)-mean)/std).float();horizon=int(rng.integers(0,8))
        with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
            for index in range(horizon):
                _,detail=model(context,torch.cat((history,future[:,index:index+1]),1),memory,True);base=detail["base"].clamp(0,1);context=torch.cat((context[:,1:],base[:,None]),1);history=torch.cat((history[:,1:],future[:,index:index+1]),1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):_,detail=model(context,torch.cat((history,future[:,horizon:horizon+1]),1),memory,True)
        truth=target[:,horizon];warped=detail["warped"];warped_hp=highpass(warped.flatten(0,1)).reshape_as(warped);truth_hp=highpass(truth)
        error=(warped-truth[:,None]).abs().mean(2)+2*(warped_hp-truth_hp[:,None]).abs().mean(2);best_source=error.argmin(1);base_hp_error=(highpass(detail["base"].detach())-truth_hp).abs().mean(1);need=(base_hp_error/(truth_hp.abs().mean(1)+1/255)).clamp(0,4).detach()
        ce=F.cross_entropy(detail["source_logits"].float(),best_source,reduction="none");loss=(ce*(1+need)).mean();accuracy=(detail["source_logits"].argmax(1)==best_source).float().mean();selected_error=error.gather(1,detail["source_logits"].argmax(1,keepdim=True)).mean();oracle_error=error.min(1).values.mean()
        if not torch.isfinite(loss):raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward();gradient=float(torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],1));optimizer.step()
        if step==1 or step%20==0:print(json.dumps({"step":step,"horizon":horizon+1,"loss":float(loss),"accuracy":float(accuracy),"selected_error":float(selected_error),"oracle_error":float(oracle_error),"gradient":gradient,"peak_memory_gib":torch.cuda.max_memory_allocated()/2**30}),flush=True)
        if step%args.validation_interval==0 or step==args.steps:
            metrics=evaluate(dev_loader,model,device,mean,std,propagate_memory=False);model.train();record={"step":step,"metrics":metrics};history_log.append(record);print(json.dumps(record),flush=True);score=min(min(v["improvement_percent"],v["texture_improvement_percent"]) for v in metrics["arms"].values());metadata={"format":"track2-v19.8-single-step-selector-training","data_boundary":"supplied_50_episodes_only","step":step,"parameters":parameter_count(),"trainable_parameters":sum(v.numel() for v in model.parameters() if v.requires_grad),"train_windows":len(train),"dev_windows":len(dev),"single_step_selector":True,"decoupled_validation":True,"history":history_log,"metrics":metrics};save(output/"latest",model,mean,std,metadata)
            if score>best:best=score;save(output/"best",model,mean,std,metadata)


if __name__=="__main__":main()
