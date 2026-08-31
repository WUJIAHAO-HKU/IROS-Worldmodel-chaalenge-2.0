#!/usr/bin/env python3
"""Measure the upper bound of target-aware source/gate selection on held-out data."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from wam_pipeline.autoregressive_texture_memory_v190 import OneStepActionTextureMemoryUNet
from train_autoregressive_unet import WindowDataset,frames_for_model


def highpass(value):return value-F.avg_pool2d(value,5,1,2,count_include_pad=False)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument("--windows",required=True);p.add_argument("--split-manifest",required=True);p.add_argument("--checkpoint",required=True);p.add_argument("--output",required=True);p.add_argument("--source-expert",action="store_true");args=p.parse_args()
    device=torch.device("cuda");split=json.loads(Path(args.split_manifest).read_text());data=WindowDataset(Path(args.windows),split["validation_episodes"]);loader=DataLoader(data,batch_size=1,num_workers=2)
    checkpoint=Path(args.checkpoint);state=torch.load(checkpoint/"model.pt",map_location="cpu",weights_only=True);model=OneStepActionTextureMemoryUNet(use_source_expert=args.source_expert).to(device)
    missing,unexpected=model.load_state_dict(state["state_dict"],strict=False)
    if unexpected or any(not key.startswith("benefit_head") for key in missing):raise ValueError((missing,unexpected))
    norm=np.load(checkpoint/"action_normalization.npz");mean=torch.from_numpy(norm["mean"]).to(device);std=torch.from_numpy(norm["std"]).to(device)
    totals={name:0. for name in ("parent_rgb","selected_rgb","best5_rgb","proxy_rgb","oracle_selected_rgb","oracle_best5_rgb","oracle_proxy_rgb","parent_texture","oracle_selected_texture","oracle_best5_texture","oracle_proxy_texture")};pixels=0
    for context,history,future,target in loader:
        context=frames_for_model(context).to(device);memory=context.clone();target=frames_for_model(target).to(device);history=((history.to(device)-mean)/std).float();future=((future.to(device)-mean)/std).float()
        for index,action in enumerate(future.unbind(1)):
            _,detail=model(context,torch.cat((history,action[:,None]),1),memory,True);base=detail["base"].clamp(0,1);truth=target[:,index];selected=detail["selected"].clamp(0,1);warped=detail["warped"].clamp(0,1)
            source_error=(warped-truth[:,None]).abs().mean(2);best_index=source_error.argmin(1);best5=warped.gather(1,best_index[:,None,None].expand(-1,1,3,-1,-1)).squeeze(1)
            warped_hp=highpass(warped.flatten(0,1)).reshape_as(warped);base_hp_for_proxy=highpass(base)
            proxy_distance=(warped-base[:,None]).abs().mean(2)+2*(warped_hp-base_hp_for_proxy[:,None]).abs().mean(2)
            proxy_index=proxy_distance.argmin(1);proxy=warped.gather(1,proxy_index[:,None,None].expand(-1,1,3,-1,-1)).squeeze(1)
            base_error=(base-truth).abs().mean(1);selected_error=(selected-truth).abs().mean(1);best_error=(best5-truth).abs().mean(1)
            proxy_error=(proxy-truth).abs().mean(1);oracle_selected=torch.where((selected_error<base_error)[:,None],selected,base);oracle_best5=torch.where((best_error<base_error)[:,None],best5,base);oracle_proxy=torch.where((proxy_error<base_error)[:,None],proxy,base)
            parent_hp=highpass(base);truth_hp=highpass(truth);selected_hp=highpass(selected);best_hp=highpass(best5)
            oracle_selected_hp=torch.where(((selected_hp-truth_hp).abs().mean(1)<(parent_hp-truth_hp).abs().mean(1))[:,None],selected_hp,parent_hp)
            oracle_best_hp=torch.where(((best_hp-truth_hp).abs().mean(1)<(parent_hp-truth_hp).abs().mean(1))[:,None],best_hp,parent_hp)
            proxy_hp=highpass(proxy);oracle_proxy_hp=torch.where(((proxy_hp-truth_hp).abs().mean(1)<(parent_hp-truth_hp).abs().mean(1))[:,None],proxy_hp,parent_hp)
            values={"parent_rgb":(base-truth).abs(),"selected_rgb":(selected-truth).abs(),"best5_rgb":(best5-truth).abs(),"proxy_rgb":(proxy-truth).abs(),"oracle_selected_rgb":(oracle_selected-truth).abs(),"oracle_best5_rgb":(oracle_best5-truth).abs(),"oracle_proxy_rgb":(oracle_proxy-truth).abs(),"parent_texture":(parent_hp-truth_hp).abs(),"oracle_selected_texture":(oracle_selected_hp-truth_hp).abs(),"oracle_best5_texture":(oracle_best_hp-truth_hp).abs(),"oracle_proxy_texture":(oracle_proxy_hp-truth_hp).abs()}
            for name,value in values.items():totals[name]+=float(value.sum())
            pixels+=truth.numel()
            context=torch.cat((context[:,1:],base[:,None]),1);history=torch.cat((history[:,1:],action[:,None]),1)
    metrics={name:255*value/pixels for name,value in totals.items()};parent=metrics["parent_rgb"];parent_texture=metrics["parent_texture"]
    for name in ("oracle_selected_rgb","oracle_best5_rgb","oracle_proxy_rgb"):metrics[name+"_improvement_percent"]=100*(parent-metrics[name])/parent
    for name in ("oracle_selected_texture","oracle_best5_texture","oracle_proxy_texture"):metrics[name+"_improvement_percent"]=100*(parent_texture-metrics[name])/parent_texture
    metrics.update({"format":"v19.8-target-aware-oracle-upper-bound" if args.source_expert else "v19.2-target-aware-oracle-upper-bound","episodes":split["validation_episodes"],"windows":len(data),"source_expert":args.source_expert})
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(metrics,indent=2)+"\n");print(json.dumps(metrics))


if __name__=="__main__":main()
