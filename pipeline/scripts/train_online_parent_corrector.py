#!/usr/bin/env python3
"""Train a zero-initialized online corrector over a frozen Track 2 parent."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,Subset,WeightedRandomSampler
from wam_pipeline.autoregressive_unet import OneStepActionUNet
from wam_pipeline.online_parent_corrector import OnlineParentCorrector
class Data(Dataset):
 def __init__(self,root,episodes):allowed=set(episodes);self.paths=[p for p in sorted(root.glob("episode*_*.npz")) if int(p.name.split("_")[0][7:]) in allowed]
 def __len__(self):return len(self.paths)
 def __getitem__(self,i):
  with np.load(self.paths[i],allow_pickle=False) as x:return tuple(torch.from_numpy(x[k].copy()) for k in ("context_frames","history_actions","future_actions","target_frames"))
def frames(x,device):return x.to(device,non_blocking=True).permute(0,1,4,2,3).float().div(255)
def hp(x):
 f=x.flatten(0,1);return (f-F.avg_pool2d(f,5,1,2,count_include_pad=False)).unflatten(0,x.shape[:2])
def rollout(parent,corrector,context,history,future):
 outputs=[];corrections=[]
 for action in future.unbind(1):
  conditioned=torch.cat((history,action[:,None]),1)
  with torch.no_grad():proposal=parent(context,conditioned).clamp(0,1)
  prediction,correction=corrector(context,proposal,conditioned);outputs.append(prediction);corrections.append(correction)
  context=torch.cat((context[:,1:].detach(),prediction[:,None].detach()),1);history=torch.cat((history[:,1:],action[:,None]),1)
 return torch.stack(outputs,1),torch.stack(corrections,1)
def loss_fn(pred,target,context,correction):
 prev=torch.cat((context[:,-1:],target[:,:-1]),1);motion=(target-prev).abs().mean(2,keepdim=True);h=torch.linspace(.7,1.3,8,device=target.device)[None,:,None,None,None];w=h*(1+2*(motion>=.03).to(target.dtype));e=pred-target;pixel=(w*e.abs()).mean()+.05*(w*e.square()).mean();texture=(w*(hp(pred)-hp(target)).abs()).mean();ex=((pred[...,1:]-pred[...,:-1])-(target[...,1:]-target[...,:-1])).abs().mean();ey=((pred[...,1:,:]-pred[...,:-1,:])-(target[...,1:,:]-target[...,:-1,:])).abs().mean();pp=torch.cat((context[:,-1:],pred[:,:-1]),1);temporal=((pred-pp)-(target-prev)).abs().mean();dark=torch.sigmoid((.32-target.mean(2,keepdim=True))*24);dl=(dark*e.abs()).sum()/(dark.sum()*3).clamp_min(1);total=pixel+.25*texture+.15*(ex+ey)/2+.15*temporal+.1*dl+.005*correction.abs().mean();return total,{"pixel":pixel,"texture":texture,"edge":(ex+ey)/2,"temporal":temporal,"dark":dl,"correction":correction.abs().mean()}
@torch.inference_mode()
def evaluate(loader,parent,model,device,mean,std):
 sums={k:0. for k in ("rgb","highpass","dark","moving")};counts={k:0 for k in sums};by=np.zeros(8);batches=0;model.eval()
 for context,history,future,target in loader:
  context,target=frames(context,device),frames(target,device);history=(history.to(device).float()-mean)/std;future=(future.to(device).float()-mean)/std
  with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):pred,_=rollout(parent,model,context,history,future)
  pred=pred.float();e=(pred-target).abs();sums["rgb"]+=float(e.sum());counts["rgb"]+=e.numel();by+=e.mean((0,2,3,4)).cpu().numpy();batches+=1;he=(hp(pred)-hp(target)).abs();sums["highpass"]+=float(he.sum());counts["highpass"]+=he.numel();prev=torch.cat((context[:,-1:],target[:,:-1]),1);moving=(target-prev).abs().mean(2,keepdim=True)>=.03;dark=target.mean(2,keepdim=True)<.32
  for k,m in (("dark",dark),("moving",moving)):m=m.expand_as(e);sums[k]+=float(e[m].sum());counts[k]+=int(m.sum())
 result={k:sums[k]/max(counts[k],1)*255 for k in sums};result["rgb_by_horizon"]=(by/batches*255).tolist();return result
def save(path,model,normalization,config,step,metrics):
 path.mkdir(parents=True,exist_ok=True);tmp=path/f"model.pt.tmp.{os.getpid()}";torch.save({"format":"track2-online-parent-corrector-v1","state_dict":model.state_dict(),"step":step,"metrics":metrics,"base_channels":model.base_channels,"residual_scale":model.residual_scale},tmp);os.replace(tmp,path/"model.pt");import shutil;shutil.copy2(normalization,path/"action_normalization.npz");(path/"training_manifest.json").write_text(json.dumps(config|{"step":step,"metrics":metrics},indent=2)+"\n")
def main():
 p=argparse.ArgumentParser();p.add_argument("--windows",required=True);p.add_argument("--split-manifest",required=True);p.add_argument("--parent",required=True);p.add_argument("--output",required=True);p.add_argument("--steps",type=int,default=2000);p.add_argument("--batch-size",type=int,default=2);p.add_argument("--learning-rate",type=float,default=5e-5);p.add_argument("--validation-interval",type=int,default=100);p.add_argument("--validation-samples",type=int,default=128);p.add_argument("--device",default="cuda");a=p.parse_args();torch.manual_seed(20260807);root=Path(a.windows);split=json.loads(Path(a.split_manifest).read_text());train=Data(root,split["train_episodes"]);valid=Data(root,split["validation_episodes"]);valid=Subset(valid,np.linspace(0,len(valid)-1,min(a.validation_samples,len(valid)),dtype=int).tolist());motion=[]
 for path in train.paths:
  with np.load(path,allow_pickle=False) as x:t=x["target_frames"].astype(np.float32);pr=np.concatenate((x["context_frames"][-1:].astype(np.float32),t[:-1]));motion.append(np.abs(t-pr).mean()/255)
 sampler=WeightedRandomSampler(torch.from_numpy(np.where(np.asarray(motion)>=.04,3.,1.)),len(train),replacement=True,generator=torch.Generator().manual_seed(20260807));common={"batch_size":a.batch_size,"num_workers":3,"pin_memory":True};train_loader=DataLoader(train,sampler=sampler,drop_last=True,**common);valid_loader=DataLoader(valid,shuffle=False,**common);device=torch.device(a.device);parent=OneStepActionUNet();raw=torch.load(Path(a.parent)/"model.pt",map_location="cpu",weights_only=True);parent.load_state_dict(raw["state_dict"]);parent.to(device).eval().requires_grad_(False);model=OnlineParentCorrector().to(device);norm=Path(a.parent)/"action_normalization.npz";z=np.load(norm);mean=torch.from_numpy(z["mean"].astype(np.float32)).to(device);std=torch.from_numpy(z["std"].astype(np.float32)).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=1e-4);config={**vars(a),"train_samples":len(train),"validation_samples":len(valid),"parameters":sum(p.numel() for p in model.parameters())};output=Path(a.output);metrics=evaluate(valid_loader,parent,model,device,mean,std);best=metrics["rgb"];save(output/"best",model,norm,config,0,metrics);print(json.dumps({"step":0,**metrics}),flush=True);iterator=iter(train_loader)
 for step in range(1,a.steps+1):
  model.train()
  try:context,history,future,target=next(iterator)
  except StopIteration:iterator=iter(train_loader);context,history,future,target=next(iterator)
  context,target=frames(context,device),frames(target,device);history=(history.to(device).float()-mean)/std;future=(future.to(device).float()-mean)/std;optimizer.zero_grad(set_to_none=True)
  with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):pred,correction=rollout(parent,model,context,history,future);loss,parts=loss_fn(pred,target,context,correction)
  loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
  if step==1 or step%25==0:print(json.dumps({"step":step,"loss":float(loss),**{k:float(v) for k,v in parts.items()}}),flush=True)
  if step%a.validation_interval==0 or step==a.steps:
   metrics=evaluate(valid_loader,parent,model,device,mean,std);print(json.dumps({"validation_step":step,**metrics}),flush=True)
   if metrics["rgb"]<best:best=metrics["rgb"];save(output/"best",model,norm,config,step,metrics)
 (output/"summary.json").write_text(json.dumps(config|{"best_rgb":best},indent=2)+"\n")
if __name__=="__main__":main()
