#!/usr/bin/env python3
"""Encode a prediction cache into deterministic per-window SDXL latents."""
import argparse,json,os
from pathlib import Path
import numpy as np,torch
from diffusers import AutoencoderKL
def main():
 p=argparse.ArgumentParser();p.add_argument("--cache",required=True);p.add_argument("--vae",required=True);p.add_argument("--output",required=True);p.add_argument("--batch-windows",type=int,default=8);p.add_argument("--device",default="cuda");a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 with np.load(a.cache,allow_pickle=False) as z:prediction=z["prediction"];names=z["windows"].astype(str)
 device=torch.device(a.device);vae=AutoencoderKL.from_pretrained(a.vae).to(device).eval();scale=float(vae.config.scaling_factor)
 with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
  for start in range(0,len(names),a.batch_windows):
   ns=names[start:start+a.batch_windows];pending=[i for i,n in enumerate(ns) if not (out/f"{Path(n).stem}.npy").exists()]
   if pending:
    x=torch.from_numpy(prediction[start:start+len(ns)][pending].copy()).to(device).permute(0,1,4,2,3).float().div(127.5).sub(1);b,t=x.shape[:2];latent=vae.encode(x.flatten(0,1)).latent_dist.mode().mul(scale).unflatten(0,(b,t)).float().cpu().numpy().astype(np.float16)
    for i,value in zip(pending,latent):
     path=out/f"{Path(ns[i]).stem}.npy";tmp=path.with_suffix(f".npy.tmp.{os.getpid()}")
     with tmp.open("wb") as h:np.save(h,value,allow_pickle=False)
     os.replace(tmp,path)
   if start%80==0 or start+len(ns)==len(names):print(json.dumps({"completed":start+len(ns),"total":len(names)}),flush=True)
 (out/"manifest.json").write_text(json.dumps({"format":"prediction-sdxl-latents-v1","cache":str(Path(a.cache).resolve()),"samples":len(names),"scaling_factor":scale},indent=2)+"\n")
if __name__=="__main__":main()
