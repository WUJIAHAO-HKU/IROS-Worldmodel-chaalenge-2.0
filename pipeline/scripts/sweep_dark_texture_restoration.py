#!/usr/bin/env python3
"""Held-out sweep for deterministic dark-structure and fine-text restoration."""

from __future__ import annotations

import argparse, json, re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def episode(name): return int(re.fullmatch(r"episode(\d+)_\d+\.npz",name).group(1))
def hp(x): return x-F.avg_pool2d(x,3,1,1)


def restore(value,gamma,sharp):
    luminance=value.mean(1,keepdim=True); gate=torch.sigmoid((.55-luminance)*16)
    darkened=value.clamp(1e-4,1).pow(gamma)
    result=value+gate*(darkened-value)
    detail=result-F.avg_pool2d(result,3,1,1)
    return (result+sharp*gate*detail).clamp(0,1)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--cache",required=True); p.add_argument("--windows",required=True); p.add_argument("--output",required=True); p.add_argument("--device",default="cuda"); a=p.parse_args()
    with np.load(a.cache,allow_pickle=False) as c: predictions=c["prediction"]; names=[str(x) for x in c["windows"]]
    dev=set(sorted(set(episode(x) for x in names))[-4:]); indices=[i for i,x in enumerate(names) if episode(x) in dev]
    candidates=[(g,s) for g in (1.,1.02,1.05,1.08,1.12,1.18,1.25) for s in (0.,.1,.2,.35,.5)]; sums={(g,s):np.zeros(4,np.float64) for g,s in candidates}; counts=np.zeros(2,np.int64); device=torch.device(a.device)
    for index in indices:
        with np.load(Path(a.windows)/names[index],allow_pickle=False) as x: target=torch.from_numpy(x["target_frames"].copy()).permute(0,3,1,2).to(device).float()/255
        parent=torch.from_numpy(predictions[index].copy()).permute(0,3,1,2).to(device).float()/255; dark=(target.mean(1,keepdim=True)<.28).expand_as(target); counts+=np.array((target.numel(),int(dark.sum())))
        target_hp=hp(target)
        for key in candidates:
            pred=restore(parent,*key); error=(pred-target).abs(); sums[key]+=np.array((float(error.sum()),float((hp(pred)-target_hp).abs().sum()),float(error[dark].sum()),float(((pred[...,1:]-pred[...,:-1])-(target[...,1:]-target[...,:-1])).abs().sum())))
    records=[]
    for (gamma,sharp),v in sums.items():
        records.append({"gamma":gamma,"sharp":sharp,"rgb":v[0]/counts[0]*255,"highpass":v[1]/counts[0]*255,"dark":v[2]/counts[1]*255,"edge_x":v[3]/(counts[0]/target.shape[-1]*(target.shape[-1]-1))*255})
    baseline=next(x for x in records if x["gamma"]==1 and x["sharp"]==0); best=min(records,key=lambda x:.35*x["rgb"]/baseline["rgb"]+.3*x["highpass"]/baseline["highpass"]+.25*x["dark"]/baseline["dark"]+.1*x["edge_x"]/baseline["edge_x"])
    result={"format":"track2-dark-texture-restoration-sweep-v1","sample_count":len(indices),"dev_episodes":sorted(dev),"baseline":baseline,"best":best,"relative_improvement_percent":{k:(baseline[k]-best[k])/baseline[k]*100 for k in ("rgb","highpass","dark","edge_x")},"candidates":records}; Path(a.output).write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({k:result[k] for k in ("sample_count","baseline","best","relative_improvement_percent")},indent=2))


if __name__=="__main__": main()
