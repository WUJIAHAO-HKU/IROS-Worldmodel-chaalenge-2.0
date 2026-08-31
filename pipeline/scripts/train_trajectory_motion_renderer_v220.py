#!/usr/bin/env python3
"""Train an action-pose-conditioned discrete motion renderer."""

from __future__ import annotations
import argparse,json,os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Subset,WeightedRandomSampler

from wam_pipeline.autoregressive_texture_memory_v190 import OneStepActionTextureMemoryUNet
from wam_pipeline.object_geometry_v170 import ActionPoseProjector,normalize_action,denormalize_pose
from wam_pipeline.trajectory_motion_renderer_v220 import BETA_LEVELS,TrajectoryMotionRenderer,decode_motion,render_motion,parameter_count
from train_autoregressive_texture_memory_v190 import highpass,window_arm_labels
from train_autoregressive_unet import WindowDataset,frames_for_model,window_motion_scores
from train_trajectory_selector_v210 import parent_trajectory


def load_pose(path,device):
    state=torch.load(path,map_location="cpu",weights_only=False);model=ActionPoseProjector(state["hidden"]).to(device);model.load_state_dict(state["model"]);model.eval().requires_grad_(False);stats=tuple(torch.from_numpy(state[k]).to(device) for k in ("action_mean","action_std","target_mean","target_std"));return model,stats


@torch.no_grad()
def project_future(pose_model,stats,raw_history,raw_future):
    sequence=torch.cat((raw_history[:,-1:],raw_future),1);motion=sequence.diff(dim=1).abs().mean(1);arms=(motion[:,7:].mean(1)>motion[:,:7].mean(1)).long();batch,time=raw_future.shape[:2];arm_time=arms[:,None].expand(-1,time);selected=torch.where(arm_time[:,:,None]==0,raw_future[:,:,:7],raw_future[:,:,7:]);flat=selected.flatten(0,1);flat_arm=arm_time.flatten();prediction=pose_model(normalize_action(flat,stats[0],stats[1],flat_arm),flat_arm);prediction=denormalize_pose(prediction,stats[2],stats[3],flat_arm).reshape(batch,time,7);return prediction[:,:,:6]/128-1,arms


@torch.no_grad()
def motion_labels(base,warped,target,context_last,minimum_gain=.00015,texture_weight=.25,temporal_weight=.25):
    blur=lambda x:F.avg_pool2d(x,9,1,4,count_include_pad=False);base_flat=base.flatten(0,1);target_flat=target.flatten(0,1);base_blur=blur(base_flat);target_hp=highpass(target_flat).reshape_as(target);previous_base=torch.cat((context_last[:,None],base[:,:-1]),1);previous_target=torch.cat((context_last[:,None],target[:,:-1]),1);costs=[]
    def cost(candidate):
        rgb=(candidate-target).abs().mean(2);hp=(highpass(candidate.flatten(0,1)).reshape_as(candidate)-target_hp).abs().mean(2);temporal=((candidate-previous_base)-(target-previous_target)).abs().mean(2);value=rgb+texture_weight*hp+temporal_weight*temporal;return F.avg_pool2d(value.flatten(0,1)[:,None],8,8).reshape(*value.shape[:2],32,32)
    costs.append(cost(base))
    for source in range(5):
        residual=(blur(warped[:,:,source].flatten(0,1))-base_blur).reshape_as(base)
        for beta in BETA_LEVELS:costs.append(cost((base+beta*residual).clamp(0,1)))
    stack=torch.stack(costs,2);best_cost,labels=stack.min(2);gain=stack[:,:,0]-best_cost;return torch.where(gain>=minimum_gain,labels,torch.zeros_like(labels))


@torch.inference_mode()
def evaluate(loader,parent,pose_model,pose_stats,renderer,device,mean,std):
    renderer.eval();pe=ve=pt=vt=0.;pdelta=vdelta=0.;accuracy=nonzero=beta_mean=0.;count=0
    for context,history,future,target in loader:
        raw_history=history.to(device);raw_future=future.to(device);context=frames_for_model(context).to(device);target=frames_for_model(target).to(device);projected,_=project_future(pose_model,pose_stats,raw_history,raw_future);history=((raw_history-mean)/std).float();future=((raw_future-mean)/std).float();base,warped=parent_trajectory(parent,context,history,future,context.clone());labels=motion_labels(base,warped,target,context[:,-1]);logits=renderer(base.float(),warped.float(),future.float(),projected.float());value,beta=render_motion(base.float(),warped.float(),logits);pe+=float((base-target).abs().mean());ve+=float((value-target).abs().mean());pt+=float((highpass(base.flatten(0,1))-highpass(target.flatten(0,1))).abs().mean());vt+=float((highpass(value.flatten(0,1))-highpass(target.flatten(0,1))).abs().mean());truth_delta=torch.cat((context[:,-1:,],target),1).diff(dim=1);pdelta+=float((torch.cat((context[:,-1:],base),1).diff(dim=1)-truth_delta).abs().mean());vdelta+=float((torch.cat((context[:,-1:],value),1).diff(dim=1)-truth_delta).abs().mean());prediction=decode_motion(logits);accuracy+=float((prediction==labels).float().mean());nonzero+=float((prediction>0).float().mean());beta_mean+=float(beta.mean());count+=1
    return {"windows":count,"label_accuracy":accuracy/count,"nonzero_patch_rate":nonzero/count,"mean_beta":beta_mean/count,"parent_rgb_mae":255*pe/count,"rgb_mae":255*ve/count,"rgb_improvement_percent":100*(pe-ve)/pe,"parent_texture_mae":255*pt/count,"texture_mae":255*vt/count,"texture_improvement_percent":100*(pt-vt)/pt,"parent_temporal_delta_mae":255*pdelta/count,"temporal_delta_mae":255*vdelta/count,"temporal_improvement_percent":100*(pdelta-vdelta)/pdelta}


def save(path,model,metadata):
    path.mkdir(parents=True,exist_ok=True);tmp=path/"motion_renderer.pt.tmp";torch.save({"format":"track2-trajectory-motion-renderer-v22.0","state_dict":model.state_dict()},tmp);os.replace(tmp,path/"motion_renderer.pt");(path/"training_manifest.json").write_text(json.dumps(metadata,indent=2)+"\n")


def main():
    p=argparse.ArgumentParser();p.add_argument("--windows",required=True);p.add_argument("--split-manifest",required=True);p.add_argument("--init-checkpoint",required=True);p.add_argument("--memory-checkpoint",required=True);p.add_argument("--pose-checkpoint",required=True);p.add_argument("--output",required=True);p.add_argument("--steps",type=int,default=400);p.add_argument("--batch-size",type=int,default=8);p.add_argument("--learning-rate",type=float,default=3e-4);p.add_argument("--validation-interval",type=int,default=50);p.add_argument("--validation-windows",type=int,default=16);p.add_argument("--seed",type=int,default=20260808);args=p.parse_args();torch.manual_seed(args.seed);np.random.seed(args.seed);device=torch.device("cuda");split=json.loads(Path(args.split_manifest).read_text());train=WindowDataset(Path(args.windows),split["train_episodes"]);dev=WindowDataset(Path(args.windows),split["validation_episodes"]);norm=np.load(Path(args.init_checkpoint)/"action_normalization.npz");mean=torch.from_numpy(norm["mean"]).to(device);std=torch.from_numpy(norm["std"]).to(device);motion=window_motion_scores(train);arms=window_arm_labels(train);counts=np.bincount(arms,minlength=2);weights=(1+3*(motion>=.03))*(len(arms)/(2*counts[arms]));sampler=WeightedRandomSampler(torch.from_numpy(weights),len(train),replacement=True,generator=torch.Generator().manual_seed(args.seed));loader=DataLoader(train,batch_size=args.batch_size,sampler=sampler,num_workers=2,pin_memory=True);iterator=iter(loader);positions=np.linspace(0,len(dev)-1,min(len(dev),args.validation_windows),dtype=int).tolist();dev_loader=DataLoader(Subset(dev,positions),batch_size=1,num_workers=2)
    parent=OneStepActionTextureMemoryUNet().to(device);state=torch.load(Path(args.init_checkpoint)/"model.pt",map_location="cpu",weights_only=True);parent.load_state_dict(state["state_dict"],strict=False);parent.initialize_memory_encoder();memory=torch.load(Path(args.memory_checkpoint)/"model.pt",map_location="cpu",weights_only=True);parent.load_state_dict(memory["state_dict"],strict=False);parent.eval().requires_grad_(False);pose_model,pose_stats=load_pose(args.pose_checkpoint,device);renderer=TrajectoryMotionRenderer().to(device);optimizer=torch.optim.AdamW(renderer.parameters(),lr=args.learning_rate,weight_decay=1e-4);output=Path(args.output);history_log=[];best=-1e9
    for step in range(1,args.steps+1):
        try:context,history,future,target=next(iterator)
        except StopIteration:iterator=iter(loader);context,history,future,target=next(iterator)
        raw_history=history.to(device);raw_future=future.to(device);context=frames_for_model(context).to(device);target=frames_for_model(target).to(device);projected,_=project_future(pose_model,pose_stats,raw_history,raw_future);history=((raw_history-mean)/std).float();future=((raw_future-mean)/std).float();base,warped=parent_trajectory(parent,context,history,future,context.clone());labels=motion_labels(base,warped,target,context[:,-1]);optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            logits=renderer(base.float(),warped.float(),future.float(),projected.float());active=labels>0
            benefit_loss=F.cross_entropy(logits[:,:,:2].permute(0,2,1,3,4).float(),active.long())
            positive_logits=logits[:,:,2:].permute(0,2,1,3,4).float();positive_target=(labels-1).clamp_min(0)
            positive_loss=F.cross_entropy(positive_logits,positive_target,reduction="none")[active].mean() if active.any() else benefit_loss.new_zeros(())
            loss=benefit_loss+positive_loss
        loss.backward();gradient=float(torch.nn.utils.clip_grad_norm_(renderer.parameters(),1));optimizer.step();prediction=decode_motion(logits)
        if step==1 or step%20==0:print(json.dumps({"step":step,"loss":float(loss),"accuracy":float((prediction==labels).float().mean()),"predicted_nonzero":float((prediction>0).float().mean()),"target_nonzero":float((labels>0).float().mean()),"gradient":gradient,"peak_memory_gib":torch.cuda.max_memory_allocated()/2**30}),flush=True)
        if step%args.validation_interval==0 or step==args.steps:
            metrics=evaluate(dev_loader,parent,pose_model,pose_stats,renderer,device,mean,std);renderer.train();record={"step":step,"metrics":metrics};history_log.append(record);print(json.dumps(record),flush=True);metadata={"format":"track2-v22.0-trajectory-motion-training","data_boundary":"supplied_50_episodes_only","step":step,"parameters":parameter_count(),"beta_levels":BETA_LEVELS,"train_windows":len(train),"dev_windows":len(dev),"history":history_log,"metrics":metrics};save(output/f"step_{step:04d}",renderer,metadata);save(output/"latest",renderer,metadata);score=min(metrics["rgb_improvement_percent"],metrics["temporal_improvement_percent"])
            if score>best:best=score;save(output/"best",renderer,metadata)


if __name__=="__main__":main()
