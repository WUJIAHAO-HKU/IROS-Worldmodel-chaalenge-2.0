#!/usr/bin/env python3
"""Train a parent-initialized direct eight-frame RGB predictor for Track 2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from wam_pipeline.direct_horizon_unet import DirectHorizonActionUNet, load_one_step_parent


class Windows(Dataset):
    def __init__(self, root: Path, episodes: list[int]):
        allowed = set(episodes)
        self.paths = [p for p in sorted(root.glob("episode*_*.npz")) if int(p.name.split("_")[0][7:]) in allowed]
    def __len__(self): return len(self.paths)
    def __getitem__(self, index):
        with np.load(self.paths[index], allow_pickle=False) as x:
            return tuple(torch.from_numpy(x[key].copy()) for key in ("context_frames", "history_actions", "future_actions", "target_frames"))


def frames(value, device): return value.to(device, non_blocking=True).permute(0, 1, 4, 2, 3).float().div(255.)


def statistics(dataset):
    values=[]; motion=[]
    for path in dataset.paths:
        with np.load(path,allow_pickle=False) as x:
            values.append(np.concatenate((x["history_actions"],x["future_actions"])))
            target=x["target_frames"].astype(np.float32)/255.; previous=np.concatenate((x["context_frames"][-1:].astype(np.float32)/255.,target[:-1]))
            motion.append(np.abs(target-previous).mean())
    actions=np.concatenate(values).astype(np.float64)
    return actions.mean(0).astype(np.float32),np.maximum(actions.std(0),1e-6).astype(np.float32),np.asarray(motion)


def hp(x):
    flat=x.flatten(0,1); return (flat-F.avg_pool2d(flat,5,1,2,count_include_pad=False)).unflatten(0,x.shape[:2])


def loss_fn(prediction,target,context):
    previous=torch.cat((context[:,-1:],target[:,:-1]),1); motion=(target-previous).abs().mean(2,keepdim=True)
    horizon=torch.linspace(.65,1.35,8,device=target.device,dtype=target.dtype)[None,:,None,None,None]
    weight=horizon*(1.+2.5*(motion>=.03).to(target.dtype))
    error=prediction-target
    pixel=(weight*error.abs()).mean()+.05*(weight*error.square()).mean()
    texture=(weight*(hp(prediction)-hp(target)).abs()).mean()
    edge_x=(weight[...,1:]*((prediction[...,1:]-prediction[...,:-1])-(target[...,1:]-target[...,:-1])).abs()).mean()
    edge_y=(weight[...,1:,:]*((prediction[...,1:,:]-prediction[...,:-1,:])-(target[...,1:,:]-target[...,:-1,:])).abs()).mean()
    predicted_previous=torch.cat((context[:,-1:],prediction[:,:-1]),1)
    temporal=(weight*((prediction-predicted_previous)-(target-previous)).abs()).mean()
    dark=torch.sigmoid((.32-target.mean(2,keepdim=True))*24.)
    dark_loss=(weight*dark*error.abs()).sum()/(weight.mul(dark).sum()*3).clamp_min(1.)
    total=pixel+.30*texture+.15*(edge_x+edge_y)/2+.20*temporal+.10*dark_loss
    return total,{"pixel":pixel,"texture":texture,"edge":(edge_x+edge_y)/2,"temporal":temporal,"dark":dark_loss}


@torch.inference_mode()
def evaluate(loader,model,device,mean,std):
    sums={k:0. for k in ("rgb","highpass","dark","moving_rgb","temporal")}; counts={k:0 for k in sums}; by=np.zeros(8); by_count=0
    model.eval()
    for context,history,future,target in loader:
        context,target=frames(context,device),frames(target,device); history=(history.to(device).float()-mean)/std; future=(future.to(device).float()-mean)/std
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):prediction=model(context,history,future).clamp(0,1)
        prediction=prediction.float(); error=(prediction-target).abs(); previous=torch.cat((context[:,-1:],target[:,:-1]),1); moving=(target-previous).abs().mean(2,keepdim=True)>=.03; dark=target.mean(2,keepdim=True)<.32
        predicted_previous=torch.cat((context[:,-1:],prediction[:,:-1]),1); temporal=((prediction-predicted_previous)-(target-previous)).abs()
        sums["rgb"]+=float(error.sum()); counts["rgb"]+=target.numel(); by+=error.mean((0,2,3,4)).cpu().numpy(); by_count+=1
        hpe=(hp(prediction)-hp(target)).abs(); sums["highpass"]+=float(hpe.sum()); counts["highpass"]+=target.numel()
        for key,mask,value in (("dark",dark,error),("moving_rgb",moving,error),("temporal",moving,temporal)):
            expanded=mask.expand_as(value); sums[key]+=float(value[expanded].sum()); counts[key]+=int(expanded.sum())
    result={k:sums[k]/max(counts[k],1)*255. for k in sums}; result["rgb_by_horizon"]=(by/by_count*255.).tolist(); return result


def save(path,model,mean,std,config,step,metrics):
    path.mkdir(parents=True,exist_ok=True); tmp=path/f"model.pt.tmp.{os.getpid()}"; torch.save({"format":"track2-direct-horizon-unet-v1","state_dict":model.state_dict(),"step":step,"metrics":metrics},tmp); os.replace(tmp,path/"model.pt")
    np.savez(path/"action_normalization.npz",mean=mean.cpu().numpy(),std=std.cpu().numpy()); (path/"training_manifest.json").write_text(json.dumps(config|{"step":step,"metrics":metrics},indent=2)+"\n")


def main():
    p=argparse.ArgumentParser(); p.add_argument("--windows",required=True); p.add_argument("--split-manifest",required=True); p.add_argument("--parent",required=True); p.add_argument("--output",required=True); p.add_argument("--steps",type=int,default=5000); p.add_argument("--batch-size",type=int,default=8); p.add_argument("--learning-rate",type=float,default=1e-4); p.add_argument("--validation-interval",type=int,default=250); p.add_argument("--validation-samples",type=int,default=128); p.add_argument("--seed",type=int,default=20260807); p.add_argument("--device",default="cuda"); a=p.parse_args()
    split=json.loads(Path(a.split_manifest).read_text()); root=Path(a.windows); train=Windows(root,split["train_episodes"]); validation=Windows(root,split["validation_episodes"]); indices=np.linspace(0,len(validation)-1,min(a.validation_samples,len(validation)),dtype=int); validation=Subset(validation,indices.tolist())
    mean_np,std_np,motion=statistics(train); weights=np.where(motion>=.04,3.,1.); sampler=WeightedRandomSampler(torch.from_numpy(weights),len(weights),replacement=True,generator=torch.Generator().manual_seed(a.seed))
    common={"batch_size":a.batch_size,"num_workers":4,"pin_memory":True}; train_loader=DataLoader(train,sampler=sampler,drop_last=True,**common); validation_loader=DataLoader(validation,shuffle=False,**common)
    device=torch.device(a.device); mean=torch.from_numpy(mean_np).to(device); std=torch.from_numpy(std_np).to(device); torch.manual_seed(a.seed); model=DirectHorizonActionUNet(); raw=torch.load(Path(a.parent)/"model.pt",map_location="cpu",weights_only=True); load_one_step_parent(model,raw["state_dict"]); model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=1e-4); output=Path(a.output); config={"format":"track2-direct-horizon-unet-v1","windows":str(root.resolve()),"split_manifest":str(Path(a.split_manifest).resolve()),"parent":str(Path(a.parent).resolve()),"steps":a.steps,"batch_size":a.batch_size,"learning_rate":a.learning_rate,"train_samples":len(train),"validation_samples":len(validation)}
    initial=evaluate(validation_loader,model,device,mean,std); best=initial["rgb"]; save(output/"best",model,mean,std,config,0,initial); print(json.dumps({"step":0,**initial}),flush=True)
    iterator=iter(train_loader)
    for step in range(1,a.steps+1):
        model.train()
        try:batch=next(iterator)
        except StopIteration:iterator=iter(train_loader);batch=next(iterator)
        context,history,future,target=batch; context,target=frames(context,device),frames(target,device); history=(history.to(device).float()-mean)/std; future=(future.to(device).float()-mean)/std
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"): prediction=model(context,history,future); loss,components=loss_fn(prediction,target,context)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
        if step==1 or step%50==0:print(json.dumps({"step":step,"loss":float(loss),**{k:float(v) for k,v in components.items()}}),flush=True)
        if step%a.validation_interval==0 or step==a.steps:
            metrics=evaluate(validation_loader,model,device,mean,std); print(json.dumps({"step":step,**metrics}),flush=True)
            if metrics["rgb"]<best:best=metrics["rgb"];save(output/"best",model,mean,std,config,step,metrics)
    (output/"summary.json").write_text(json.dumps(config|{"best_rgb":best},indent=2)+"\n")
if __name__=="__main__":main()
