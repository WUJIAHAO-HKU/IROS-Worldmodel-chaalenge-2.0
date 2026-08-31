#!/usr/bin/env python3
"""Train a discrete full-trajectory benefit renderer on the supplied episodes."""

from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from wam_pipeline.autoregressive_texture_memory_v190 import OneStepActionTextureMemoryUNet
from wam_pipeline.trajectory_texture_selector_v210 import TrajectoryPatchSelector
from wam_pipeline.trajectory_benefit_renderer_v212 import ALPHA_LEVELS, TrajectoryBenefitRenderer, render, parameter_count
from train_autoregressive_texture_memory_v190 import highpass, window_arm_labels
from train_autoregressive_unet import WindowDataset, frames_for_model, window_motion_scores
from train_trajectory_selector_v210 import parent_trajectory


@torch.no_grad()
def select_sources(selector, base, warped, actions):
    logits = selector(base.float(), warped.float(), actions.float())
    probabilities = logits.softmax(2)
    source = probabilities.argmax(2)
    patch = F.one_hot(source, 5).permute(0, 1, 4, 2, 3).float()
    weights = F.interpolate(patch.flatten(0, 1), size=base.shape[-2:], mode="nearest")
    weights = weights.reshape(len(base), 8, 5, *base.shape[-2:])
    selected = (warped * weights[:, :, :, None]).sum(2)
    return selected, probabilities


@torch.no_grad()
def alpha_labels(base, selected, target, minimum_gain=.00015, texture_weight=2.):
    levels = torch.tensor(ALPHA_LEVELS, device=base.device, dtype=base.dtype)
    base_hp = highpass(base.flatten(0, 1)).reshape_as(base)
    selected_hp = highpass(selected.flatten(0, 1)).reshape_as(selected)
    candidates = (base[:, :, None] + levels[None, None, :, None, None, None]
                  * (selected_hp - base_hp)[:, :, None]).clamp(0, 1)
    target_hp = highpass(target.flatten(0, 1)).reshape_as(target)
    candidate_hp = highpass(candidates.flatten(0, 2)).reshape_as(candidates)
    cost = (candidates - target[:, :, None]).abs().mean(3)
    cost += texture_weight * (candidate_hp - target_hp[:, :, None]).abs().mean(3)
    patch_cost = F.avg_pool2d(cost.flatten(0, 2), 8, 8).reshape(*cost.shape[:3], 32, 32)
    best_cost, labels = patch_cost.min(2)
    labels = torch.where(patch_cost[:, :, 0] - best_cost >= minimum_gain, labels, torch.zeros_like(labels))
    labels[:, :3] = 0
    return labels


@torch.inference_mode()
def evaluate(loader, parent, selector, renderer, device, mean, std, texture_weight):
    renderer.eval(); parent_error = value_error = parent_texture = value_texture = 0.; accuracy = nonzero = alpha_mean = 0.; count = 0
    for context, history, future, target in loader:
        context = frames_for_model(context).to(device); target = frames_for_model(target).to(device)
        history = ((history.to(device) - mean) / std).float(); future = ((future.to(device) - mean) / std).float()
        base, warped = parent_trajectory(parent, context, history, future, context.clone())
        selected, probabilities = select_sources(selector, base, warped, future)
        labels = alpha_labels(base, selected, target, texture_weight=texture_weight)
        logits = renderer(base.float(), selected.float(), probabilities.float(), future.float())
        value, alpha = render(base.float(), selected.float(), logits)
        parent_error += float((base - target).abs().mean()) * len(base); value_error += float((value - target).abs().mean()) * len(base)
        parent_texture += float((highpass(base.flatten(0, 1)) - highpass(target.flatten(0, 1))).abs().mean()) * len(base)
        value_texture += float((highpass(value.flatten(0, 1)) - highpass(target.flatten(0, 1))).abs().mean()) * len(base)
        accuracy += float((logits.argmax(2) == labels).float().mean()) * len(base)
        nonzero += float((logits.argmax(2) != 0).float().mean()) * len(base); alpha_mean += float(alpha.mean()) * len(base); count += len(base)
    return {"sample_count": count, "label_accuracy": accuracy/count, "nonzero_patch_rate": nonzero/count,
            "mean_alpha": alpha_mean/count, "parent_rgb_mae": 255*parent_error/count, "rgb_mae": 255*value_error/count,
            "rgb_improvement_percent": 100*(parent_error-value_error)/parent_error,
            "parent_texture_mae": 255*parent_texture/count, "texture_mae": 255*value_texture/count,
            "texture_improvement_percent": 100*(parent_texture-value_texture)/parent_texture}


def save(path, renderer, metadata):
    path.mkdir(parents=True, exist_ok=True); temporary = path / "renderer.pt.tmp"
    torch.save({"format": "track2-trajectory-benefit-renderer-v21.2", "state_dict": renderer.state_dict()}, temporary)
    os.replace(temporary, path / "renderer.pt"); (path / "training_manifest.json").write_text(json.dumps(metadata, indent=2)+"\n")


def main():
    p=argparse.ArgumentParser(); p.add_argument("--windows",required=True);p.add_argument("--split-manifest",required=True);p.add_argument("--init-checkpoint",required=True);p.add_argument("--memory-checkpoint",required=True);p.add_argument("--selector-checkpoint",required=True);p.add_argument("--output",required=True);p.add_argument("--steps",type=int,default=400);p.add_argument("--batch-size",type=int,default=8);p.add_argument("--learning-rate",type=float,default=2e-4);p.add_argument("--validation-interval",type=int,default=50);p.add_argument("--validation-windows",type=int,default=16);p.add_argument("--texture-weight",type=float,default=2.);p.add_argument("--seed",type=int,default=20260808);args=p.parse_args()
    torch.manual_seed(args.seed);np.random.seed(args.seed);device=torch.device("cuda");split=json.loads(Path(args.split_manifest).read_text());train=WindowDataset(Path(args.windows),split["train_episodes"]);dev=WindowDataset(Path(args.windows),split["validation_episodes"])
    norm=np.load(Path(args.init_checkpoint)/"action_normalization.npz");mean=torch.from_numpy(norm["mean"]).to(device);std=torch.from_numpy(norm["std"]).to(device);motion=window_motion_scores(train);arms=window_arm_labels(train);counts=np.bincount(arms,minlength=2);weights=(1+3*(motion>=.03))*(len(arms)/(2*counts[arms]));sampler=WeightedRandomSampler(torch.from_numpy(weights),len(train),replacement=True,generator=torch.Generator().manual_seed(args.seed));loader=DataLoader(train,batch_size=args.batch_size,sampler=sampler,num_workers=2,pin_memory=True);iterator=iter(loader);positions=np.linspace(0,len(dev)-1,min(len(dev),args.validation_windows),dtype=int).tolist();dev_loader=DataLoader(Subset(dev,positions),batch_size=1,num_workers=2)
    parent=OneStepActionTextureMemoryUNet().to(device);initial=torch.load(Path(args.init_checkpoint)/"model.pt",map_location="cpu",weights_only=True);parent.load_state_dict(initial["state_dict"],strict=False);parent.initialize_memory_encoder();memory=torch.load(Path(args.memory_checkpoint)/"model.pt",map_location="cpu",weights_only=True);parent.load_state_dict(memory["state_dict"],strict=False);parent.eval().requires_grad_(False)
    selector=TrajectoryPatchSelector().to(device);selector.load_state_dict(torch.load(Path(args.selector_checkpoint)/"selector.pt",map_location="cpu",weights_only=True)["state_dict"]);selector.eval().requires_grad_(False);renderer=TrajectoryBenefitRenderer().to(device);optimizer=torch.optim.AdamW(renderer.parameters(),lr=args.learning_rate,weight_decay=1e-4);output=Path(args.output);history_log=[];best=-1e9
    for step in range(1,args.steps+1):
        try:context,history,future,target=next(iterator)
        except StopIteration:iterator=iter(loader);context,history,future,target=next(iterator)
        context=frames_for_model(context).to(device);target=frames_for_model(target).to(device);history=((history.to(device)-mean)/std).float();future=((future.to(device)-mean)/std).float();base,warped=parent_trajectory(parent,context,history,future,context.clone());selected,probabilities=select_sources(selector,base,warped,future);labels=alpha_labels(base,selected,target,texture_weight=args.texture_weight);optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):logits=renderer(base.float(),selected.float(),probabilities.float(),future.float());loss=F.cross_entropy(logits.permute(0,2,1,3,4).float(),labels)
        loss.backward();gradient=float(torch.nn.utils.clip_grad_norm_(renderer.parameters(),1));optimizer.step();prediction=logits.argmax(2);accuracy=float((prediction==labels).float().mean());nonzero=float((prediction!=0).float().mean())
        if step==1 or step%20==0:print(json.dumps({"step":step,"loss":float(loss),"accuracy":accuracy,"predicted_nonzero":nonzero,"target_nonzero":float((labels!=0).float().mean()),"gradient":gradient,"peak_memory_gib":torch.cuda.max_memory_allocated()/2**30}),flush=True)
        if step%args.validation_interval==0 or step==args.steps:
            metrics=evaluate(dev_loader,parent,selector,renderer,device,mean,std,args.texture_weight);renderer.train();record={"step":step,"metrics":metrics};history_log.append(record);print(json.dumps(record),flush=True);metadata={"format":"track2-v21.2-trajectory-benefit-renderer-training","data_boundary":"supplied_50_episodes_only","step":step,"parameters":parameter_count(),"alpha_levels":ALPHA_LEVELS,"texture_label_weight":args.texture_weight,"suppress_first":3,"train_windows":len(train),"dev_windows":len(dev),"history":history_log,"metrics":metrics};save(output/f"step_{step:04d}",renderer,metadata);save(output/"latest",renderer,metadata)
            score=min(metrics["rgb_improvement_percent"],metrics["texture_improvement_percent"])
            if score>best:best=score;save(output/"best",renderer,metadata)


if __name__=="__main__":main()
