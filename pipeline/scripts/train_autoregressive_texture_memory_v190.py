#!/usr/bin/env python3
"""Train persistent texture memory inside the frozen autoregressive parent."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from wam_pipeline.autoregressive_texture_memory_v190 import OneStepActionTextureMemoryUNet, parameter_count
from train_autoregressive_unet import (
    WindowDataset, action_statistics, frames_for_model, reconstruction_loss, window_motion_scores,
)


def highpass(value):
    return value-F.avg_pool2d(value,5,1,2,count_include_pad=False)


def window_arm_labels(dataset):
    labels=[]
    for path in dataset.paths:
        with np.load(path,allow_pickle=False) as data:
            sequence=np.concatenate((data["history_actions"][-1:],data["future_actions"]),0)
        delta=np.abs(np.diff(sequence,axis=0)).mean(0)
        labels.append(int(delta[7:].mean()>delta[:7].mean()))
    return np.asarray(labels,dtype=np.int64)


def rollout(model, context, history, future, memory, enable_memory=True, propagate_memory=True, suppress_first=0):
    predictions=[]; diagnostics=[]
    for frame_index,action in enumerate(future.unbind(1)):
        candidate, detail=model(context,torch.cat((history,action[:,None]),1),memory,True)
        frame=(candidate if enable_memory and frame_index>=suppress_first else detail["base"]).clamp(0,1); predictions.append(frame); diagnostics.append(detail)
        state_frame=frame if propagate_memory else detail["base"].clamp(0,1)
        context=torch.cat((context[:,1:],state_frame[:,None]),1); history=torch.cat((history[:,1:],action[:,None]),1)
    return torch.stack(predictions,1),diagnostics


@torch.inference_mode()
def evaluate(loader,model,device,mean,std,propagate_memory=True,suppress_first=0):
    model.eval(); parent_sum=torch.zeros(8); value_sum=torch.zeros(8); count=0
    parent_texture_sum=torch.zeros(8);value_texture_sum=torch.zeros(8)
    arm_values={0:[0.,0.,0.,0.,0],1:[0.,0.,0.,0.,0]}
    for context,history,future,target in loader:
        context=frames_for_model(context).to(device); memory=context.clone(); target=frames_for_model(target).to(device)
        raw_history=history.clone(); raw_future=future.clone()
        history=((history.to(device)-mean)/std).float(); future=((future.to(device)-mean)/std).float()
        value,_=rollout(model,context,history,future,memory,propagate_memory=propagate_memory,suppress_first=suppress_first)
        # Parent branch is available exactly before the memory correction.
        parent_context=context.clone(); parent_history=history.clone(); parents=[]
        for action in future.unbind(1):
            _,detail=model(parent_context,torch.cat((parent_history,action[:,None]),1),memory,True)
            base=detail["base"].clamp(0,1); parents.append(base)
            parent_context=torch.cat((parent_context[:,1:],base[:,None]),1); parent_history=torch.cat((parent_history[:,1:],action[:,None]),1)
        parent=torch.stack(parents,1)
        pe=(parent-target).abs().mean((2,3,4)).cpu(); ve=(value-target).abs().mean((2,3,4)).cpu()
        pte=(highpass(parent.flatten(0,1))-highpass(target.flatten(0,1))).abs().mean((1,2,3)).reshape(len(context),8).cpu()
        vte=(highpass(value.flatten(0,1))-highpass(target.flatten(0,1))).abs().mean((1,2,3)).reshape(len(context),8).cpu()
        parent_sum+=pe.sum(0); value_sum+=ve.sum(0); count+=len(context)
        parent_texture_sum+=pte.sum(0);value_texture_sum+=vte.sum(0)
        action_sequence=torch.cat((raw_history[:,-1:],raw_future),1)
        action_delta=action_sequence.diff(dim=1).abs().mean(1)
        arms=(action_delta[:,7:].mean(1)>action_delta[:,:7].mean(1)).long()
        for i,arm in enumerate(arms.tolist()):
            arm_values[arm][0]+=float(pe[i].mean());arm_values[arm][1]+=float(ve[i].mean())
            arm_values[arm][2]+=float(pte[i].mean());arm_values[arm][3]+=float(vte[i].mean());arm_values[arm][4]+=1
    p=parent_sum/count*255;v=value_sum/count*255;pt=parent_texture_sum/count*255;vt=value_texture_sum/count*255
    result={"sample_count":count,"parent_rgb_mae":float(p.mean()),"memory_rgb_mae":float(v.mean()),
            "improvement_percent":100*float((p.mean()-v.mean())/p.mean()),
            "parent_texture_mae":float(pt.mean()),"memory_texture_mae":float(vt.mean()),
            "texture_improvement_percent":100*float((pt.mean()-vt.mean())/pt.mean()),
            "parent_frame_mae":p.tolist(),"memory_frame_mae":v.tolist(),"arms":{}}
    for arm,(ps,vs,pts,vts,n) in arm_values.items():
        if not n: continue
        pm=255*ps/n;vm=255*vs/n;ptm=255*pts/n;vtm=255*vts/n
        result["arms"][f"arm{arm}"]={"sample_count":n,"parent_rgb_mae":pm,"memory_rgb_mae":vm,"improvement_percent":100*(pm-vm)/pm,
                                         "parent_texture_mae":ptm,"memory_texture_mae":vtm,"texture_improvement_percent":100*(ptm-vtm)/ptm}
    return result


def save(path,model,mean,std,metadata):
    path.mkdir(parents=True,exist_ok=True); tmp=path/"model.pt.tmp"
    torch.save({"format":"track2-autoregressive-texture-memory-v19.5","state_dict":model.state_dict()},tmp);os.replace(tmp,path/"model.pt")
    np.savez(path/"action_normalization.npz",mean=mean.cpu().numpy(),std=std.cpu().numpy())
    (path/"training_manifest.json").write_text(json.dumps(metadata,indent=2)+"\n")


def main():
    p=argparse.ArgumentParser();p.add_argument("--windows",required=True);p.add_argument("--split-manifest",required=True)
    p.add_argument("--init-checkpoint",required=True);p.add_argument("--output",required=True);p.add_argument("--steps",type=int,default=500)
    p.add_argument("--batch-size",type=int,default=1);p.add_argument("--learning-rate",type=float,default=3e-4)
    p.add_argument("--validation-interval",type=int,default=100);p.add_argument("--validation-windows",type=int,default=16)
    p.add_argument("--alignment-steps",type=int,default=400)
    p.add_argument("--memory-checkpoint");p.add_argument("--gate-only",action="store_true")
    p.add_argument("--arm-balanced",action="store_true");p.add_argument("--gate-margin",type=float,default=.25)
    p.add_argument("--seed",type=int,default=20260808);p.add_argument("--device",default="cuda");args=p.parse_args()
    torch.manual_seed(args.seed);np.random.seed(args.seed);device=torch.device(args.device);split=json.loads(Path(args.split_manifest).read_text())
    train=WindowDataset(Path(args.windows),split["train_episodes"]);dev=WindowDataset(Path(args.windows),split["validation_episodes"])
    normalization=np.load(Path(args.init_checkpoint)/"action_normalization.npz")
    mean=torch.from_numpy(normalization["mean"]).to(device);std=torch.from_numpy(normalization["std"]).to(device);scores=window_motion_scores(train)
    weights=1+3*(scores>=.03)
    if args.arm_balanced:
        labels=window_arm_labels(train);counts=np.bincount(labels,minlength=2)
        weights=weights*(len(labels)/(2*counts[labels]))
    sampler=WeightedRandomSampler(torch.from_numpy(weights),len(train),replacement=True,generator=torch.Generator().manual_seed(args.seed))
    loader=DataLoader(train,batch_size=args.batch_size,sampler=sampler,num_workers=2,pin_memory=True)
    positions=np.linspace(0,len(dev)-1,min(len(dev),args.validation_windows),dtype=int).tolist();dev_loader=DataLoader(Subset(dev,positions),batch_size=1,num_workers=2)
    model=OneStepActionTextureMemoryUNet().to(device);state=torch.load(Path(args.init_checkpoint)/"model.pt",map_location="cpu",weights_only=True)
    if state.get("format")!="track2-autoregressive-unet-v1":raise ValueError("parent checkpoint format")
    missing,unexpected=model.load_state_dict(state["state_dict"],strict=False)
    if unexpected or any(not key.startswith(("memory_enc0","memory_fusion","memory_head","benefit_head","benefit_expert")) for key in missing):raise ValueError((missing,unexpected))
    model.initialize_memory_encoder()
    if args.memory_checkpoint:
        memory_state=torch.load(Path(args.memory_checkpoint)/"model.pt",map_location="cpu",weights_only=True)
        memory_missing,memory_unexpected=model.load_state_dict(memory_state["state_dict"],strict=False)
        if memory_unexpected or any(not key.startswith(("benefit_head","benefit_expert")) for key in memory_missing):raise ValueError((memory_missing,memory_unexpected))
    trainable=("benefit_expert",) if args.gate_only else ("memory_enc0","memory_fusion","memory_head","benefit_head","benefit_expert")
    for name,value in model.named_parameters():value.requires_grad=name.startswith(trainable)
    optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=args.learning_rate,weight_decay=1e-4)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True);history=[];best=-1e9;iterator=iter(loader)
    for step in range(1,args.steps+1):
        alignment_phase=(not args.gate_only) and step<=args.alignment_steps
        if not args.gate_only and step==args.alignment_steps+1:
            for group in optimizer.param_groups:group["lr"]=args.learning_rate/3
        try:context,history_action,future,target=next(iterator)
        except StopIteration:iterator=iter(loader);context,history_action,future,target=next(iterator)
        context=frames_for_model(context).to(device);memory=context.clone();target=frames_for_model(target).to(device)
        history_action=((history_action.to(device)-mean)/std).float();future=((future.to(device)-mean)/std).float()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
            prediction,details=rollout(model,context,history_action,future,memory,enable_memory=not alignment_phase);previous=torch.cat((context[:,-1:],target[:,:-1]),1)
            reconstruction=reconstruction_loss(prediction,target,previous,motion_weight=4,motion_threshold=.02,horizon_loss_power=1,
                                                temporal_delta_weight=.25,texture_laplacian_weight=.75,temporal_delta_pool=1)
            alignment=prediction.new_zeros(());source_loss=prediction.new_zeros(());gate_supervision=prediction.new_zeros(());flow_smoothness=prediction.new_zeros(())
            for frame_index,detail in enumerate(details):
                truth=target[:,frame_index];truth_hp=highpass(truth);warped=detail["warped"]
                warped_hp=highpass(warped.flatten(0,1)).reshape_as(warped)
                per_source=(warped-truth[:,None]).abs().mean(2)+2*(warped_hp-truth_hp[:,None]).abs().mean(2)
                best_error,best_source=per_source.min(1)
                base_error=(detail["base"]-truth).abs().mean(1)+2*(highpass(detail["base"])-truth_hp).abs().mean(1)
                need=((highpass(detail["base"])-truth_hp).abs().mean(1)/(truth_hp.abs().mean(1)+1/255)).clamp(0,4).detach()
                alignment=alignment+((1+2*need)*best_error).mean()/8
                source_loss=source_loss+(F.cross_entropy(detail["source_logits"],best_source,reduction="none")*(1+need)).mean()/8
                selected=detail["selected"];selected_hp=highpass(selected)
                selected_rgb_error=(selected-truth).abs().mean(1);base_rgb_error=(detail["base"]-truth).abs().mean(1)
                selected_hp_error=(selected_hp-truth_hp).abs().mean(1);base_hp_error=(highpass(detail["base"])-truth_hp).abs().mean(1)
                texture_delta=(selected_hp-highpass(detail["base"])).detach().float();structure_delta=(model.structure_scale*(selected-detail["base"])).detach().float();residual=(truth-detail["base"]).detach().float()
                tt=texture_delta.square().sum(1)+1e-3;ss=structure_delta.square().sum(1)+1e-3;ts=(texture_delta*structure_delta).sum(1)
                tr=(texture_delta*residual).sum(1);sr=(structure_delta*residual).sum(1);det=(tt*ss-ts.square()).clamp_min(1e-6)
                texture_oracle=((tr*ss-sr*ts)/det).nan_to_num().clamp(0,1).to(prediction.dtype);structure_oracle=((sr*tt-tr*ts)/det).nan_to_num().clamp(0,1).to(prediction.dtype)
                texture_oracle=torch.where(texture_oracle>.02,texture_oracle,torch.zeros_like(texture_oracle));structure_oracle=torch.where(structure_oracle>.02,structure_oracle,torch.zeros_like(structure_oracle))
                gate_logits=torch.cat((detail["texture_logit"],detail["structure_logit"]),1)
                gate_target=torch.stack((texture_oracle,structure_oracle),1)
                gate_supervision=gate_supervision+F.binary_cross_entropy_with_logits(gate_logits,gate_target)/8
                flow=detail["flow"]
                flow_smoothness=flow_smoothness+((flow[:,:,:,1:]-flow[:,:,:,:-1]).abs().mean()+(flow[:,:,:,:,1:]-flow[:,:,:,:,:-1]).abs().mean())/8
            accurate=(details[0]["base"]-target[:,0]).abs().mean(1,keepdim=True)<(2/255)
            protect=(prediction[:,0]-details[0]["base"]).abs()[accurate.expand_as(prediction[:,0])].mean() if accurate.any() else reconstruction.new_zeros(())
            gate=torch.stack([d["texture_gate"].mean() for d in details]).mean()
            if alignment_phase:
                loss=.5*alignment+.02*source_loss+.01*flow_smoothness
            elif args.gate_only:
                loss=reconstruction+.05*gate_supervision+.1*protect
            else:
                loss=reconstruction+.25*alignment+.01*source_loss+.05*gate_supervision+.01*flow_smoothness+.1*protect
        if not torch.isfinite(loss):raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward();torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],1);optimizer.step()
        if step==1 or step%20==0:
            memory_gib=torch.cuda.max_memory_allocated()/2**30 if device.type=="cuda" else 0
            print(json.dumps({"step":step,"phase":"alignment" if alignment_phase else "gating","loss":float(loss),"reconstruction":float(reconstruction),"alignment":float(alignment),"source":float(source_loss),"gate_supervision":float(gate_supervision),"mae":float((prediction-target).abs().mean()),"texture_gate":float(gate),"protect":float(protect),"peak_memory_gib":memory_gib}),flush=True)
        if step%args.validation_interval==0 or step==args.steps:
            metrics=evaluate(dev_loader,model,device,mean,std);model.train();record={"step":step,"metrics":metrics};history.append(record);print(json.dumps(record),flush=True)
            score=min(min(v["improvement_percent"],v["texture_improvement_percent"]) for v in metrics["arms"].values())
            metadata={"format":"track2-autoregressive-texture-memory-v19.5-training","data_boundary":"supplied_50_episodes_only","step":step,"phase":"alignment" if alignment_phase else "gating","parameters":parameter_count(),"trainable_parameters":sum(v.numel() for v in model.parameters() if v.requires_grad),"train_windows":len(train),"dev_windows":len(dev),"frozen_parent":True,"direct_alignment_supervision":True,"continuous_optimal_gate_supervision":True,"alignment_steps":args.alignment_steps,"gate_only":args.gate_only,"arm_balanced":args.arm_balanced,"gate_margin":args.gate_margin,"memory_checkpoint":args.memory_checkpoint,"history":history,"metrics":metrics}
            save(output/"latest",model,mean,std,metadata)
            if score>best:best=score;save(output/"best",model,mean,std,metadata)


if __name__=="__main__":main()
