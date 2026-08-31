#!/usr/bin/env python3
"""Train an unconstrained full-frame texture/structure restoration parent."""

from __future__ import annotations

import argparse, json, os, re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from wam_pipeline.global_texture_structure_refiner import GlobalTextureStructureRefiner


def episode(name): return int(re.fullmatch(r"episode(\d+)_\d+\.npz", name).group(1))


class Data(Dataset):
    def __init__(self, windows, prediction, names, indices): self.windows,self.prediction,self.names,self.indices=windows,prediction,names,indices
    def __len__(self): return len(self.indices)
    def __getitem__(self,item):
        index=self.indices[item]
        with np.load(self.windows/self.names[index],allow_pickle=False) as x:
            return x["context_frames"].copy(),self.prediction[index],np.concatenate((x["history_actions"],x["future_actions"])).copy(),x["target_frames"].copy()


def images(x,device): return x.permute(0,1,4,2,3).to(device,non_blocking=True).float().div(255)
def highpass(x):
    flat=x.flatten(0,1); return (flat-F.avg_pool2d(flat,5,1,2,count_include_pad=False)).unflatten(0,x.shape[:2])


def loss_fn(pred,target,parent,context,correction):
    error=(pred-target).abs(); target_hp=highpass(target); hp=(highpass(pred)-target_hp).abs(); dark=torch.sigmoid((.38-target.mean(2,keepdim=True))*28); detail=(target_hp.abs().mean(2,keepdim=True)/.06).clamp(0,3); motion=(target-torch.cat((context[:,-1:],target[:,:-1]),1)).abs().mean(2,keepdim=True); parent_error=(parent-target).abs().mean(2,keepdim=True); weight=1+3*dark+3*detail+2*(motion/.04).clamp(0,3)+6*(parent_error/.05).clamp(0,4)
    edge_x=((pred[...,1:]-pred[...,:-1])-(target[...,1:]-target[...,:-1])).abs(); edge_y=((pred[...,1:,:]-pred[...,:-1,:])-(target[...,1:,:]-target[...,:-1,:])).abs()
    pixel=(weight*error).mean(); texture=(weight*hp).mean(); edge=.5*(edge_x.mean()+edge_y.mean()); temporal=((pred-torch.cat((context[:,-1:],pred[:,:-1]),1))-(target-torch.cat((context[:,-1:],target[:,:-1]),1))).abs().mean(); preserve=(correction.abs()/(1+10*(parent-target).abs())).mean()
    residual_target=target-parent; residual_mse=(weight*(correction-residual_target).square()).mean(); residual_texture=(weight*(highpass(correction)-highpass(residual_target)).square()).mean(); missing_dark=(dark*(pred.mean(2,keepdim=True)-target.mean(2,keepdim=True)).relu()).mean()
    total=pixel+.35*texture+.2*edge+.12*temporal+2.0*residual_mse+1.0*residual_texture+1.0*missing_dark+.002*preserve
    return total,{"pixel":pixel,"texture":texture,"edge":edge,"temporal":temporal,"residual_mse":residual_mse,"residual_texture":residual_texture,"missing_dark":missing_dark}


@torch.inference_mode()
def evaluate(loader,model,device,mean,std):
    sums={k:0. for k in ("parent_rgb","rgb","parent_highpass","highpass","parent_dark","dark")}; counts={"all":0,"dark":0}
    model.eval()
    for context,parent,actions,target in loader:
        context,parent,target=images(context,device),images(parent,device),images(target,device); actions=(actions.to(device).float()-mean)/std
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"): pred,_,_=model(context,parent,actions)
        dark=(target.mean(2,keepdim=True)<.28).expand_as(target); pe=(parent-target).abs(); e=(pred.float()-target).abs(); sums["parent_rgb"]+=float(pe.sum()); sums["rgb"]+=float(e.sum()); sums["parent_highpass"]+=float((highpass(parent)-highpass(target)).abs().sum()); sums["highpass"]+=float((highpass(pred.float())-highpass(target)).abs().sum()); sums["parent_dark"]+=float(pe[dark].sum()); sums["dark"]+=float(e[dark].sum()); counts["all"]+=target.numel(); counts["dark"]+=int(dark.sum())
    result={"parent":{"rgb":sums["parent_rgb"]/counts["all"]*255,"highpass":sums["parent_highpass"]/counts["all"]*255,"dark":sums["parent_dark"]/counts["dark"]*255},"refiner":{"rgb":sums["rgb"]/counts["all"]*255,"highpass":sums["highpass"]/counts["all"]*255,"dark":sums["dark"]/counts["dark"]*255}}
    result["relative_improvement_percent"]={k:(result["parent"][k]-result["refiner"][k])/result["parent"][k]*100 for k in result["parent"]}; return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("--windows",required=True); p.add_argument("--cache",required=True); p.add_argument("--normalization",required=True); p.add_argument("--output",required=True); p.add_argument("--steps",type=int,default=1000); p.add_argument("--batch-size",type=int,default=2); p.add_argument("--base-channels",type=int,default=32); p.add_argument("--learning-rate",type=float,default=2e-4); p.add_argument("--validation-interval",type=int,default=100); p.add_argument("--device",default="cuda"); a=p.parse_args(); torch.manual_seed(20260808); np.random.seed(20260808)
    with np.load(a.cache,allow_pickle=False) as c: prediction=c["prediction"]; names=[str(x) for x in c["windows"]]
    eps=sorted(set(episode(x) for x in names)); dev=set(eps[-4:]); train_idx=[i for i,x in enumerate(names) if episode(x) not in dev]; dev_idx=[i for i,x in enumerate(names) if episode(x) in dev]
    train=Data(Path(a.windows),prediction,names,train_idx); valid=Data(Path(a.windows),prediction,names,dev_idx); train_loader=DataLoader(train,batch_size=a.batch_size,shuffle=True,num_workers=2,pin_memory=True,drop_last=True); valid_loader=DataLoader(valid,batch_size=a.batch_size,num_workers=2,pin_memory=True)
    n=np.load(a.normalization); device=torch.device(a.device); mean=torch.from_numpy(n["mean"].astype(np.float32)).to(device); std=torch.from_numpy(n["std"].astype(np.float32)).to(device); model=GlobalTextureStructureRefiner(a.base_channels).to(device); opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=1e-4); iterator=iter(train_loader); output=Path(a.output); output.mkdir(parents=True,exist_ok=True); history=[]
    for step in range(1,a.steps+1):
        try: context,parent,actions,target=next(iterator)
        except StopIteration: iterator=iter(train_loader); context,parent,actions,target=next(iterator)
        context,parent,target=images(context,device),images(parent,device),images(target,device); actions=(actions.to(device).float()-mean)/std; opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"): pred,correction,gate=model(context,parent,actions); loss,parts=loss_fn(pred,target,parent,context,correction)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
        if step%25==0: print(json.dumps({"step":step,"loss":float(loss),"gate":float(gate.mean()),"correction":float(correction.abs().mean()),"gpu_mb":torch.cuda.max_memory_allocated()/2**20,**{k:float(v) for k,v in parts.items()}}),flush=True)
        if step%a.validation_interval==0 or step==a.steps:
            metrics=evaluate(valid_loader,model,device,mean,std); history.append({"step":step,**metrics}); print(json.dumps(history[-1]),flush=True); model.train(); tmp=output/f"model_step_{step:06d}.pt.tmp"; torch.save({"format":"track2-global-texture-structure-refiner-v1","step":step,"state_dict":model.state_dict(),"base_channels":a.base_channels,"residual_scale":model.residual_scale,"history":history},tmp); os.replace(tmp,output/f"model_step_{step:06d}.pt")
    (output/"training_manifest.json").write_text(json.dumps({"format":"track2-global-texture-structure-refiner-v1","train":len(train_idx),"dev":len(dev_idx),"dev_episodes":sorted(dev),"history":history},indent=2)+"\n")


if __name__=="__main__": main()
