#!/usr/bin/env python3
"""Calibrate low-frequency structure strength on the full held-out development set."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from wam_pipeline.autoregressive_texture_memory_v190 import OneStepActionTextureMemoryUNet
from train_autoregressive_texture_memory_v190 import evaluate
from train_autoregressive_unet import WindowDataset


def main():
    p=argparse.ArgumentParser();p.add_argument("--windows",required=True);p.add_argument("--split-manifest",required=True)
    p.add_argument("--checkpoint",required=True);p.add_argument("--output",required=True);p.add_argument("--scales",nargs="+",type=float,default=[0,.1,.2,.35]);args=p.parse_args()
    device=torch.device("cuda");split=json.loads(Path(args.split_manifest).read_text());data=WindowDataset(Path(args.windows),split["validation_episodes"])
    loader=DataLoader(data,batch_size=1,num_workers=2,pin_memory=True);checkpoint=Path(args.checkpoint)
    state=torch.load(checkpoint/"model.pt",map_location="cpu",weights_only=True);normalization=np.load(checkpoint/"action_normalization.npz")
    mean=torch.from_numpy(normalization["mean"]).to(device);std=torch.from_numpy(normalization["std"]).to(device);results=[]
    for scale in args.scales:
        model=OneStepActionTextureMemoryUNet(structure_scale=scale).to(device)
        missing,unexpected=model.load_state_dict(state["state_dict"],strict=False)
        if unexpected or any(not key.startswith("benefit_head") for key in missing):raise ValueError((missing,unexpected))
        metrics=evaluate(loader,model,device,mean,std);metrics["structure_scale"]=scale;results.append(metrics)
        print(json.dumps({"structure_scale":scale,"rgb":metrics["improvement_percent"],"texture":metrics["texture_improvement_percent"],"arms":metrics["arms"]}),flush=True)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps({"format":"v19.3-structure-scale-sweep","episodes":split["validation_episodes"],"windows":len(data),"results":results},indent=2)+"\n")


if __name__=="__main__":main()
