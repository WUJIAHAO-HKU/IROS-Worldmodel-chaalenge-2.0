#!/usr/bin/env python3
"""Train a deterministic action-conditioned SDXL-latent future predictor."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from diffusers import AutoencoderKL
from wam_pipeline.direct_latent_unet import DirectLatentUNet

class Data(Dataset):
 def __init__(self,windows,latents,names,mean,std):self.windows,self.latents,self.names,self.mean,self.std=windows,latents,names,mean,std
 def __len__(self):return len(self.names)
 def __getitem__(self,i):
  name=self.names[i]; latent=np.load(self.latents/f"{Path(name).stem}.npy").astype(np.float32)
  with np.load(self.windows/name,allow_pickle=False) as x:history=(x["history_actions"].astype(np.float32)-self.mean)/self.std;future=(x["future_actions"].astype(np.float32)-self.mean)/self.std
  return torch.from_numpy(latent[:5]),torch.from_numpy(history),torch.from_numpy(future),torch.from_numpy(latent[5:]),name

def stats(windows,names):
 values=[];motion=[]
 for name in names:
  with np.load(windows/name,allow_pickle=False) as x:
   values.append(np.concatenate((x["history_actions"],x["future_actions"])));t=x["target_frames"].astype(np.float32);p=np.concatenate((x["context_frames"][-1:].astype(np.float32),t[:-1]));motion.append(np.abs(t-p).mean()/255.)
 a=np.concatenate(values).astype(np.float64);return a.mean(0).astype(np.float32),np.maximum(a.std(0),1e-6).astype(np.float32),np.asarray(motion)

def latent_loss(pred,target,context):
 previous=torch.cat((context[:,-1:],target[:,:-1]),1);h=torch.linspace(.7,1.3,8,device=target.device)[None,:,None,None,None];e=pred-target
 pixel=(h*e.abs()).mean()+.05*(h*e.square()).mean();spatial=((pred[...,1:]-pred[...,:-1])-(target[...,1:]-target[...,:-1])).abs().mean()+((pred[...,1:,:]-pred[...,:-1,:])-(target[...,1:,:]-target[...,:-1,:])).abs().mean();pp=torch.cat((context[:,-1:],pred[:,:-1]),1);temporal=((pred-pp)-(target-previous)).abs().mean();return pixel+.15*spatial+.2*temporal,{"latent":pixel,"spatial":spatial,"temporal":temporal}

@torch.inference_mode()
def latent_eval(loader,model,device):
 total=count=0;model.eval()
 for context,history,future,target,_ in loader:
  context,history,future,target=[v.to(device,non_blocking=True).float() for v in (context,history,future,target)]
  with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):pred=model(context,history,future)
  total+=float((pred.float()-target).abs().sum());count+=target.numel()
 return total/count

@torch.inference_mode()
def rgb_eval(loader,model,vae,scaling,windows,device):
 total=count=0;model.eval();vae.eval()
 for context,history,future,_,names in loader:
  context,history,future=[v.to(device,non_blocking=True).float() for v in (context,history,future)]
  with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):latent=model(context,history,future).flatten(0,1)/scaling;chunks=[vae.decode(latent[i:i+16]).sample for i in range(0,len(latent),16)];rgb=torch.cat(chunks).float().add(1).mul(127.5).clamp(0,255).unflatten(0,(len(names),8)).permute(0,1,3,4,2).cpu()
  target=[]
  for name in names:
   with np.load(windows/name,allow_pickle=False) as x:target.append(x["target_frames"].copy())
  y=torch.from_numpy(np.stack(target));total+=float((rgb-y).abs().sum());count+=y.numel()
 return total/count

def save(path,model,mean,std,config,step,latent_mae,rgb_mae):
 path.mkdir(parents=True,exist_ok=True);tmp=path/f"model.pt.tmp.{os.getpid()}";torch.save({"format":"track2-direct-latent-unet-v1","state_dict":model.state_dict(),"step":step,"latent_mae":latent_mae,"rgb_mae":rgb_mae,"base_channels":model.base_channels},tmp);os.replace(tmp,path/"model.pt");np.savez(path/"action_normalization.npz",mean=mean,std=std);(path/"training_manifest.json").write_text(json.dumps(config|{"step":step,"latent_mae":latent_mae,"rgb_mae":rgb_mae},indent=2)+"\n")

def main():
 p=argparse.ArgumentParser();p.add_argument("--windows",required=True);p.add_argument("--split-manifest",required=True);p.add_argument("--train-latents",required=True);p.add_argument("--validation-latents",required=True);p.add_argument("--vae",required=True);p.add_argument("--output",required=True);p.add_argument("--steps",type=int,default=10000);p.add_argument("--batch-size",type=int,default=32);p.add_argument("--learning-rate",type=float,default=2e-4);p.add_argument("--validation-interval",type=int,default=500);p.add_argument("--rgb-interval",type=int,default=1000);p.add_argument("--validation-samples",type=int,default=64);p.add_argument("--device",default="cuda");a=p.parse_args();torch.manual_seed(20260807)
 windows=Path(a.windows);split=json.loads(Path(a.split_manifest).read_text());names=lambda es:[q.name for e in es for q in sorted(windows.glob(f"episode{e}_*.npz"))];train_names=names(split["train_episodes"]);valid_names=names(split["validation_episodes"]);valid_names=[valid_names[i] for i in np.linspace(0,len(valid_names)-1,min(a.validation_samples,len(valid_names)),dtype=int)];mean,std,motion=stats(windows,train_names)
 train=Data(windows,Path(a.train_latents),train_names,mean,std);valid=Data(windows,Path(a.validation_latents),valid_names,mean,std);sampler=WeightedRandomSampler(torch.from_numpy(np.where(motion>=.04,3.,1.)),len(train),replacement=True,generator=torch.Generator().manual_seed(20260807));train_loader=DataLoader(train,batch_size=a.batch_size,sampler=sampler,num_workers=4,pin_memory=True,drop_last=True);valid_loader=DataLoader(valid,batch_size=min(8,a.batch_size),shuffle=False,num_workers=2,pin_memory=True)
 device=torch.device(a.device);model=DirectLatentUNet().to(device);vae=AutoencoderKL.from_pretrained(a.vae).to(device).eval();scaling=float(vae.config.scaling_factor);optimizer=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=1e-4);config={**vars(a),"train_samples":len(train),"validation_samples":len(valid),"parameters":sum(p.numel() for p in model.parameters())};output=Path(a.output);best=float("inf");iterator=iter(train_loader)
 initial_latent=latent_eval(valid_loader,model,device);initial_rgb=rgb_eval(valid_loader,model,vae,scaling,windows,device);print(json.dumps({"step":0,"latent_mae":initial_latent,"rgb_mae":initial_rgb,"parameters":config["parameters"]}),flush=True)
 for step in range(1,a.steps+1):
  model.train()
  try:context,history,future,target,_=next(iterator)
  except StopIteration:iterator=iter(train_loader);context,history,future,target,_=next(iterator)
  context,history,future,target=[v.to(device,non_blocking=True).float() for v in (context,history,future,target)];optimizer.zero_grad(set_to_none=True)
  with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):pred=model(context,history,future);loss,parts=latent_loss(pred,target,context)
  loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
  if step==1 or step%50==0:print(json.dumps({"step":step,"loss":float(loss),**{k:float(v) for k,v in parts.items()}}),flush=True)
  if step%a.validation_interval==0 or step==a.steps:
   lm=latent_eval(valid_loader,model,device);rm=rgb_eval(valid_loader,model,vae,scaling,windows,device) if step%a.rgb_interval==0 or step==a.steps else None;print(json.dumps({"validation_step":step,"latent_mae":lm,"rgb_mae":rm}),flush=True)
   score=rm if rm is not None else float("inf")
   if score<best:best=score;save(output/"best",model,mean,std,config,step,lm,rm)
 (output/"summary.json").write_text(json.dumps(config|{"best_rgb_mae":best},indent=2)+"\n")
if __name__=="__main__":main()
