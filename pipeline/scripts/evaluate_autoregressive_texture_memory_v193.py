#!/usr/bin/env python3
"""Evaluate a v19.3 checkpoint on every window of held-out episodes."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from wam_pipeline.autoregressive_texture_memory_v190 import OneStepActionTextureMemoryUNet, parameter_count
from train_autoregressive_texture_memory_v190 import evaluate
from train_autoregressive_unet import WindowDataset


def improvement(parent, value):
    return 100*(float(np.mean(parent))-float(np.mean(value)))/float(np.mean(parent))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--windows",required=True);parser.add_argument("--split-manifest",required=True)
    parser.add_argument("--checkpoint",required=True);parser.add_argument("--output",required=True);parser.add_argument("--device",default="cuda");parser.add_argument("--decoupled",action="store_true");parser.add_argument("--source-expert",action="store_true")
    parser.add_argument("--suppress-first",type=int,default=0)
    args=parser.parse_args();device=torch.device(args.device);split=json.loads(Path(args.split_manifest).read_text())
    dataset=WindowDataset(Path(args.windows),split["validation_episodes"]);loader=DataLoader(dataset,batch_size=1,num_workers=2,pin_memory=True)
    checkpoint=Path(args.checkpoint);state=torch.load(checkpoint/"model.pt",map_location="cpu",weights_only=True)
    model=OneStepActionTextureMemoryUNet(use_source_expert=args.source_expert).to(device);missing,unexpected=model.load_state_dict(state["state_dict"],strict=False)
    if unexpected or any(not key.startswith(("benefit_head","benefit_expert","source_expert")) for key in missing):raise ValueError((missing,unexpected))
    normalization=np.load(checkpoint/"action_normalization.npz")
    mean=torch.from_numpy(normalization["mean"]).to(device);std=torch.from_numpy(normalization["std"]).to(device)
    metrics=evaluate(loader,model,device,mean,std,propagate_memory=not args.decoupled,suppress_first=args.suppress_first);metrics.update({"format":"track2-v19.9-full-development-evaluation",
        "checkpoint":str(checkpoint),"episodes":split["validation_episodes"],"windows":len(dataset),"parameters":parameter_count(),
        "decoupled_texture_dynamics":args.decoupled,
        "source_expert":args.source_expert,
        "suppress_first":args.suppress_first,
        "early_frame_improvement_percent":improvement(metrics["parent_frame_mae"][:4],metrics["memory_frame_mae"][:4]),
        "late_frame_improvement_percent":improvement(metrics["parent_frame_mae"][4:],metrics["memory_frame_mae"][4:])})
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(metrics,indent=2)+"\n")
    print(json.dumps(metrics))


if __name__=="__main__":main()
