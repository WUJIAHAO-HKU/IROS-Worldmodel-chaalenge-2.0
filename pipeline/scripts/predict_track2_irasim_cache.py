#!/usr/bin/env python3
"""Batch-cache deterministic Track 2 rollouts from a trained IRASim adapter."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, PNDMScheduler

from wam_pipeline.irasim_lora import inject_irasim_lora


def load_pretrained(model,path):
    raw=torch.load(path,map_location="cpu",weights_only=False); state=raw.get("ema",raw); target=model.state_dict(); loaded={}
    for key,value in state.items():
        key=key.removeprefix("module.")
        if key not in target: continue
        if value.shape==target[key].shape: loaded[key]=value
        elif key=="pos_embed": loaded[key]=F.interpolate(value.reshape(1,16,20,-1).permute(0,3,1,2).float(),(16,16),mode="bicubic",align_corners=False).permute(0,2,3,1).reshape_as(target[key])
        elif key=="temp_embed": loaded[key]=F.interpolate(value.permute(0,2,1).float(),size=13,mode="linear",align_corners=False).permute(0,2,1)
        elif key=="embed_state.fc1.weight": loaded[key]=torch.cat((value,value),1)*.5
    model.load_state_dict(loaded,strict=False)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--irasim-root",required=True); p.add_argument("--pretrained",required=True); p.add_argument("--adapter",required=True); p.add_argument("--vae",required=True); p.add_argument("--windows",required=True); p.add_argument("--latents",required=True); p.add_argument("--names-cache",help="Optional prediction NPZ whose window list defines the sample population."); p.add_argument("--output",required=True); p.add_argument("--sample-count",type=int,default=64); p.add_argument("--batch-size",type=int,default=4); p.add_argument("--steps",type=int,default=50); p.add_argument("--seed",type=int,default=0); p.add_argument("--device",default="cuda"); a=p.parse_args()
    sys.path.insert(0,str(Path(a.irasim_root).resolve())); from models.irasim import IRASim_models; from sample.pipeline_trajectory2videogen import Trajectory2VideoGenPipeline
    adapter=torch.load(a.adapter,map_location="cpu",weights_only=False); cfg=SimpleNamespace(dataset="bridge",state_dim=14,final_frame_ada=False,gradient_checkpointing=False); model=IRASim_models["IRASim-XL/2"](input_size=32,num_frames=13,learn_sigma=False,extras=3,attention_mode="sdpa",args=cfg); load_pretrained(model,a.pretrained); inject_irasim_lora(model,int(adapter["config"]["lora_rank"]),float(adapter["config"]["lora_alpha"])); model.load_state_dict(adapter["state_dict"],strict=False); device=torch.device(a.device); model.to(device).eval(); vae=AutoencoderKL.from_pretrained(a.vae).to(device).eval(); scheduler=PNDMScheduler.from_pretrained(Path(a.irasim_root)/"pretrained_models/scheduler",beta_start=.0001,beta_end=.02,beta_schedule="linear",variance_type="fixed_small"); pipe=Trajectory2VideoGenPipeline(vae=vae,scheduler=scheduler,transformer=model)
    if a.names_cache:
        with np.load(a.names_cache,allow_pickle=False) as population: all_names=[Path(str(x)).stem for x in population["windows"]]
    else: all_names=sorted(x.stem for x in Path(a.latents).glob("*.npy"))
    indices=np.linspace(0,len(all_names)-1,min(a.sample_count,len(all_names)),dtype=int); names=[all_names[i] for i in indices]; predictions=[]; generator=torch.Generator(device=device).manual_seed(a.seed); mean=np.asarray(adapter["action_mean"]); std=np.asarray(adapter["action_std"])
    for start in range(0,len(names),a.batch_size):
        current=names[start:start+a.batch_size]; context=torch.from_numpy(np.stack([np.load(Path(a.latents)/f"{name}.npy")[:5] for name in current]).astype(np.float32)).to(device); actions=[]
        for name in current:
            with np.load(Path(a.windows)/f"{name}.npz",allow_pickle=False) as x: actions.append(np.concatenate((x["history_actions"],x["future_actions"])))
        action=torch.from_numpy(((np.stack(actions)-mean)/std).astype(np.float32)).to(device)
        with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"): latent,_=pipe(action,mask_x=context,video_length=13,height=256,width=256,num_inference_steps=a.steps,guidance_scale=1.,generator=generator,device=device,output_type="video"); future=latent[:,5:].flatten(0,1)/vae.config.scaling_factor; decoded=[]
        with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
            for offset in range(0,len(future),8): decoded.append(vae.decode(future[offset:offset+8]).sample)
        rgb=torch.cat(decoded).unflatten(0,(len(current),8)).float().add(1).mul(127.5).clamp(0,255).round().byte().permute(0,1,3,4,2).cpu().numpy(); predictions.append(rgb); print(json.dumps({"completed":min(start+a.batch_size,len(names)),"total":len(names),"seed":a.seed}),flush=True)
    window_names=[f"{name}.npz" for name in names]; output=Path(a.output); output.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(output,prediction=np.concatenate(predictions),windows=np.asarray(window_names)); output.with_suffix(".json").write_text(json.dumps({"format":"track2-irasim-cache-v1","seed":a.seed,"steps":a.steps,"sample_count":len(names),"windows":window_names},indent=2)+"\n")


if __name__=="__main__": main()
