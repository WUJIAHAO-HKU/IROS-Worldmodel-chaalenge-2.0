#!/usr/bin/env python3
"""Train decoded semantic deltas while retaining the parent's exact RGB detail."""
from __future__ import annotations
import argparse,json,os,re
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from diffusers import AutoencoderKL
from wam_pipeline.latent_semantic_delta_refiner import LatentSemanticDeltaRefiner

def load_cache(path):
 with np.load(path,allow_pickle=False) as z:return z["prediction"],z["windows"].astype(str).tolist()
def action_stats(windows,names):
 v=[]
 for n in names:
  with np.load(windows/n,allow_pickle=False) as x:v.append(np.concatenate((x["history_actions"],x["future_actions"])))
 a=np.concatenate(v).astype(np.float64);return a.mean(0).astype(np.float32),np.maximum(a.std(0),1e-6).astype(np.float32)
class Data(Dataset):
 def __init__(self,windows,source_latents,base_latents,base_rgb,names,mean,std):self.windows,self.source,self.base,self.rgb,self.names,self.mean,self.std=windows,source_latents,base_latents,base_rgb,names,mean,std
 def __len__(self):return len(self.names)
 def __getitem__(self,i):
  n=self.names[i];context=np.load(self.source/f"{Path(n).stem}.npy")[:5].astype(np.float32);baseline=np.load(self.base/f"{Path(n).stem}.npy").astype(np.float32)
  with np.load(self.windows/n,allow_pickle=False) as x:history=(x["history_actions"].astype(np.float32)-self.mean)/self.std;future=(x["future_actions"].astype(np.float32)-self.mean)/self.std;context_rgb=x["context_frames"][-1].copy();target=x["target_frames"].copy()
  return context,baseline,self.rgb[i].copy(),history,future,context_rgb,target
def hp(x):
 flat=x.flatten(0,1);return (flat-F.avg_pool2d(flat,5,1,2,count_include_pad=False)).unflatten(0,x.shape[:2])
def decoded_delta(model,vae,scale,context,baseline,base_rgb,history,future):
 corrected,delta=model(context,baseline,history,future);b,t=baseline.shape[:2]
 with torch.no_grad():base_dec=vae.decode(baseline.flatten(0,1)/scale).sample
 corrected_dec=vae.decode(corrected.flatten(0,1)/scale).sample
 correction=(corrected_dec-base_dec).unflatten(0,(b,t)).mul(.5);prediction=(base_rgb+correction).clamp(0,1);return prediction,delta
def loss_fn(pred,target,context_rgb,delta):
 previous=torch.cat((context_rgb[:,-1:],target[:,:-1]),1);motion=(target-previous).abs().mean(2,keepdim=True);weight=1+2*(motion>=.03).to(target.dtype);e=prediction_error=pred-target;pixel=(weight*e.abs()).mean()+.05*(weight*e.square()).mean();texture=(weight*(hp(pred)-hp(target)).abs()).mean();ex=((pred[...,1:]-pred[...,:-1])-(target[...,1:]-target[...,:-1])).abs().mean();ey=((pred[...,1:,:]-pred[...,:-1,:])-(target[...,1:,:]-target[...,:-1,:])).abs().mean();pp=torch.cat((context_rgb[:,-1:],pred[:,:-1]),1);temporal=((pred-pp)-(target-previous)).abs().mean();dark=torch.sigmoid((.32-target.mean(2,keepdim=True))*24);dark_loss=(dark*e.abs()).sum()/(dark.sum()*3).clamp_min(1);total=pixel+.25*texture+.15*(ex+ey)/2+.15*temporal+.1*dark_loss+.001*delta.abs().mean();return total,{"pixel":pixel,"texture":texture,"edge":(ex+ey)/2,"temporal":temporal,"dark":dark_loss,"delta":delta.abs().mean()}
@torch.inference_mode()
def evaluate(loader,model,vae,scale,device):
 sums={k:0. for k in ("rgb","highpass","dark","moving")};counts={k:0 for k in sums};model.eval()
 for context,baseline,base_rgb,history,future,context_rgb,target in loader:
  context,baseline,history,future=[v.to(device).float() for v in (context,baseline,history,future)];base_rgb=base_rgb.to(device).permute(0,1,4,2,3).float().div(255);target=target.to(device).permute(0,1,4,2,3).float().div(255)
  with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):pred,_=decoded_delta(model,vae,scale,context,baseline,base_rgb,history,future)
  context_rgb=context_rgb.to(device).permute(0,3,1,2).float().div(255);pred=pred.float();e=(pred-target).abs();sums["rgb"]+=float(e.sum());counts["rgb"]+=e.numel();he=(hp(pred)-hp(target)).abs();sums["highpass"]+=float(he.sum());counts["highpass"]+=he.numel();previous=torch.cat((context_rgb[:,None],target[:,:-1]),1);moving=(target-previous).abs().mean(2,keepdim=True)>=.03;dark=target.mean(2,keepdim=True)<.32
  for k,m in (("dark",dark),("moving",moving)):m=m.expand_as(e);sums[k]+=float(e[m].sum());counts[k]+=int(m.sum())
 return {k:sums[k]/max(counts[k],1)*255 for k in sums}
def save(path,model,mean,std,config,step,metrics):
 path.mkdir(parents=True,exist_ok=True);tmp=path/f"model.pt.tmp.{os.getpid()}";torch.save({"format":"track2-latent-semantic-delta-refiner-v1","state_dict":model.state_dict(),"step":step,"metrics":metrics,"base_channels":model.base_channels},tmp);os.replace(tmp,path/"model.pt");np.savez(path/"action_normalization.npz",mean=mean,std=std);(path/"training_manifest.json").write_text(json.dumps(config|{"step":step,"metrics":metrics},indent=2)+"\n")
def main():
 p=argparse.ArgumentParser();p.add_argument("--windows",required=True);p.add_argument("--train-cache",required=True);p.add_argument("--validation-cache",required=True);p.add_argument("--train-source-latents",required=True);p.add_argument("--validation-source-latents",required=True);p.add_argument("--train-base-latents",required=True);p.add_argument("--validation-base-latents",required=True);p.add_argument("--vae",required=True);p.add_argument("--output",required=True);p.add_argument("--steps",type=int,default=2500);p.add_argument("--batch-size",type=int,default=2);p.add_argument("--learning-rate",type=float,default=5e-5);p.add_argument("--validation-interval",type=int,default=100);p.add_argument("--device",default="cuda");a=p.parse_args();torch.manual_seed(20260807);windows=Path(a.windows);trgb,tn=load_cache(a.train_cache);vrgb,vn=load_cache(a.validation_cache);mean,std=action_stats(windows,tn);train=Data(windows,Path(a.train_source_latents),Path(a.train_base_latents),trgb,tn,mean,std);valid=Data(windows,Path(a.validation_source_latents),Path(a.validation_base_latents),vrgb,vn,mean,std);train_loader=DataLoader(train,batch_size=a.batch_size,shuffle=True,num_workers=2,pin_memory=True,drop_last=True,generator=torch.Generator().manual_seed(20260807));valid_loader=DataLoader(valid,batch_size=a.batch_size,shuffle=False,num_workers=2,pin_memory=True);device=torch.device(a.device);model=LatentSemanticDeltaRefiner().to(device);vae=AutoencoderKL.from_pretrained(a.vae).to(device).eval();vae.requires_grad_(False);scale=float(vae.config.scaling_factor);optimizer=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=1e-4);config={**vars(a),"train_samples":len(train),"validation_samples":len(valid),"parameters":sum(p.numel() for p in model.parameters())};output=Path(a.output);metrics=evaluate(valid_loader,model,vae,scale,device);best=metrics["rgb"];save(output/"best",model,mean,std,config,0,metrics);print(json.dumps({"step":0,**metrics}),flush=True);iterator=iter(train_loader)
 for step in range(1,a.steps+1):
  model.train()
  try:batch=next(iterator)
  except StopIteration:iterator=iter(train_loader);batch=next(iterator)
  context,baseline,base_rgb,history,future,context_rgb,target=batch;context,baseline,history,future=[v.to(device,non_blocking=True).float() for v in (context,baseline,history,future)];base_rgb=base_rgb.to(device).permute(0,1,4,2,3).float().div(255);context_rgb=context_rgb.to(device).permute(0,3,1,2).float().div(255)[:,None];target=target.to(device).permute(0,1,4,2,3).float().div(255);optimizer.zero_grad(set_to_none=True)
  with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):pred,delta=decoded_delta(model,vae,scale,context,baseline,base_rgb,history,future);loss,parts=loss_fn(pred,target,context_rgb,delta)
  loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
  if step==1 or step%25==0:print(json.dumps({"step":step,"loss":float(loss),**{k:float(v) for k,v in parts.items()}}),flush=True)
  if step%a.validation_interval==0 or step==a.steps:
   metrics=evaluate(valid_loader,model,vae,scale,device);print(json.dumps({"validation_step":step,**metrics}),flush=True)
   if metrics["rgb"]<best:best=metrics["rgb"];save(output/"best",model,mean,std,config,step,metrics)
 (output/"summary.json").write_text(json.dumps(config|{"best_rgb":best},indent=2)+"\n")
if __name__=="__main__":main()
