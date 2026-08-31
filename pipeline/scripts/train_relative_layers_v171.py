#!/usr/bin/env python3
"""Train v17.1 observed-geometry-relative object motion heads."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from wam_pipeline.object_geometry_v170 import RelativeLayerProjector


def pairs(cache: Path, episodes: set[int]) -> tuple[np.ndarray, ...]:
    with np.load(cache, allow_pickle=False) as data:
        action=data["action"]; landmark=data["target"]; arms=data["arm_id"]
        layers=data["layer_id"]; episode_ids=data["episode"]; frames=data["frame"]
    features=[]; deltas=[]; out_arms=[]; out_layers=[]; out_episodes=[]
    for episode in sorted(episodes):
        for arm in (0,1):
            for layer in range(3):
                selected=np.flatnonzero((episode_ids==episode)&(arms==arm)&(layers==layer))
                lookup={int(frames[index]):index for index in selected}
                for source_frame, source_index in lookup.items():
                    for horizon in range(1,9):
                        target_index=lookup.get(source_frame+horizon)
                        if target_index is None: continue
                        feature=np.concatenate((landmark[source_index],action[source_index],
                                                action[target_index],[horizon/8.0])).astype(np.float32)
                        delta=(landmark[target_index]-landmark[source_index]).astype(np.float32)
                        features.append(feature); deltas.append(delta); out_arms.append(arm)
                        out_layers.append(layer); out_episodes.append(episode)
    return (np.asarray(features,np.float32),np.asarray(deltas,np.float32),np.asarray(out_arms,np.int64),
            np.asarray(out_layers,np.int64),np.asarray(out_episodes,np.int64))


def stats(value,arms,layers):
    mean=np.zeros((2,3,value.shape[1]),np.float32); std=np.ones_like(mean)
    for arm in (0,1):
        for layer in range(3):
            selected=(arms==arm)&(layers==layer)
            if selected.any():
                mean[arm,layer]=value[selected].mean(0); std[arm,layer]=np.maximum(value[selected].std(0),1e-4)
    return mean,std


@torch.inference_mode()
def evaluate(model,tensors,norm,batch_size):
    feature,delta,arms,layers=tensors; fm,fs,dm,ds=norm; predicted=[]
    for start in range(0,len(feature),batch_size):
        s=slice(start,start+batch_size); a=arms[s];l=layers[s]
        value=model((feature[s]-fm[a,l])/fs[a,l],a,l)*ds[a,l]+dm[a,l]; predicted.append(value)
    error=(torch.cat(predicted)[:,:6]-delta[:,:6]).abs(); result={"sample_count":len(delta),"delta_landmark_mae_px":float(error.mean()),"groups":{}}
    for arm in (0,1):
        for layer,name in enumerate(model.layer_names):
            selected=(arms==arm)&(layers==layer)
            result["groups"][f"arm{arm}_{name}"]={"sample_count":int(selected.sum()),
                "delta_landmark_mae_px":float(error[selected].mean()) if selected.any() else None}
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--labels",required=True);p.add_argument("--split",required=True);p.add_argument("--output",required=True)
    p.add_argument("--dev-episodes",default="36,47");p.add_argument("--steps",type=int,default=5000);p.add_argument("--batch-size",type=int,default=1024)
    p.add_argument("--learning-rate",type=float,default=1e-3);p.add_argument("--hidden",type=int,default=256);p.add_argument("--seed",type=int,default=20260808)
    p.add_argument("--device",default="cuda");p.add_argument("--log-every",type=int,default=250);args=p.parse_args()
    torch.manual_seed(args.seed);np.random.seed(args.seed);output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    split=json.loads(Path(args.split).read_text());dev_eps={int(x) for x in args.dev_episodes.split(",") if x};train_eps=set(split["train_episodes"])-dev_eps
    values=pairs(Path(args.labels),train_eps|dev_eps);train_mask=np.isin(values[4],sorted(train_eps));dev_mask=np.isin(values[4],sorted(dev_eps))
    train=tuple(x[train_mask] for x in values[:4]);dev=tuple(x[dev_mask] for x in values[:4]);fm,fs=stats(train[0],train[2],train[3]);dm,ds=stats(train[1],train[2],train[3])
    device=torch.device(args.device);train_t=tuple(torch.from_numpy(x).to(device) for x in train);dev_t=tuple(torch.from_numpy(x).to(device) for x in dev)
    norm=tuple(torch.from_numpy(x).to(device) for x in (fm,fs,dm,ds));model=RelativeLayerProjector(args.hidden).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-5);gen=torch.Generator(device=device).manual_seed(args.seed)
    best=float("inf");history=[]
    for step in range(1,args.steps+1):
        index=torch.randint(len(train_t[0]),(args.batch_size,),generator=gen,device=device);feature,delta,arms,layers=(x[index] for x in train_t)
        prediction=model((feature-norm[0][arms,layers])/norm[1][arms,layers],arms,layers);target=(delta-norm[2][arms,layers])/norm[3][arms,layers]
        loss=torch.nn.functional.smooth_l1_loss(prediction[:,:6],target[:,:6])+0.1*torch.nn.functional.smooth_l1_loss(prediction[:,6],target[:,6])
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        if step%args.log_every==0 or step==args.steps:
            model.eval();metrics=evaluate(model,dev_t,norm,args.batch_size);model.train();record={"step":step,"loss":float(loss),"dev":metrics};history.append(record);print(json.dumps(record),flush=True)
            score=max(v["delta_landmark_mae_px"] for v in metrics["groups"].values() if v["delta_landmark_mae_px"] is not None)
            payload={"format":"track2-relative-layer-projector-v17.1","step":step,"hidden":args.hidden,"model":model.state_dict(),
                     "feature_mean":fm,"feature_std":fs,"delta_mean":dm,"delta_std":ds,"dev_metrics":metrics}
            torch.save(payload,output/"latest.pt")
            if score<best:best=score;torch.save(payload,output/"best.pt")
    manifest={"format":"track2-relative-layer-projector-v17.1-training","train_episodes":sorted(train_eps),"dev_episodes":sorted(dev_eps),
              "train_pairs":len(train[0]),"dev_pairs":len(dev[0]),"steps":args.steps,"parameter_count":sum(x.numel() for x in model.parameters()),
              "best_worst_group_delta_landmark_mae_px":best,"history":history}
    tmp=output/f"training_manifest.json.tmp.{os.getpid()}";tmp.write_text(json.dumps(manifest,indent=2)+"\n");os.replace(tmp,output/"training_manifest.json")


if __name__=="__main__":main()
