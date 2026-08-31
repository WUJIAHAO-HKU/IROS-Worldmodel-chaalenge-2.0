#!/usr/bin/env python3
"""Generate one Track 2 rollout from a pretrained IRASim + joint14 LoRA adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as functional
from diffusers.models import AutoencoderKL
from diffusers.schedulers import DPMSolverMultistepScheduler, PNDMScheduler

from wam_pipeline.irasim_lora import inject_irasim_lora


def highpass(value):
    flat=value.flatten(0,1)
    return value-functional.avg_pool2d(flat,3,stride=1,padding=1).view_as(value)


def load_pretrained(model, path: Path):
    raw=torch.load(path,map_location="cpu",weights_only=False); state=raw.get("ema",raw.get("model",raw)); state={k.removeprefix("module."):v for k,v in state.items()}; target=model.state_dict(); loaded={}
    for key,value in state.items():
        if key not in target: continue
        if value.shape==target[key].shape: loaded[key]=value
        elif key=="pos_embed":
            old=value.reshape(1,16,value.shape[1]//16,value.shape[-1]).permute(0,3,1,2); loaded[key]=functional.interpolate(old.float(),(16,16),mode="bicubic",align_corners=False).permute(0,2,3,1).reshape_as(target[key])
        elif key=="temp_embed": loaded[key]=functional.interpolate(value.permute(0,2,1).float(),size=13,mode="linear",align_corners=False).permute(0,2,1)
        elif key=="embed_state.fc1.weight" and value.shape[1]==7: loaded[key]=torch.cat((value,value),1)*0.5
    model.load_state_dict(loaded,strict=False)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--irasim-root",required=True); parser.add_argument("--pretrained",required=True); parser.add_argument("--adapter",required=True); parser.add_argument("--vae",required=True); parser.add_argument("--window",required=True); parser.add_argument("--output",required=True); parser.add_argument("--steps",type=int,default=50); parser.add_argument("--sampler",choices=("pndm","dpm"),default="pndm"); parser.add_argument("--initial-prediction",help="Optional [8,H,W,3] prediction used for diffusion refinement."); parser.add_argument("--strength",type=float,default=1.0); parser.add_argument("--seed",type=int,default=0); parser.add_argument("--device",default="cuda"); args=parser.parse_args()
    sys.path.insert(0,str(Path(args.irasim_root).resolve())); from models.irasim import IRASim_models; from sample.pipeline_trajectory2videogen import Trajectory2VideoGenPipeline
    config=SimpleNamespace(dataset="bridge",state_dim=14,final_frame_ada=False,gradient_checkpointing=False); model=IRASim_models["IRASim-XL/2"](input_size=32,num_frames=13,learn_sigma=False,extras=3,attention_mode="sdpa",args=config); load_pretrained(model,Path(args.pretrained)); adapter=torch.load(args.adapter,map_location="cpu",weights_only=False); rank=int(adapter["config"]["lora_rank"]); inject_irasim_lora(model,rank,float(adapter["config"]["lora_alpha"])); incompatible=model.load_state_dict(adapter["state_dict"],strict=False); unexpected=[k for k in incompatible.unexpected_keys];
    if unexpected: raise RuntimeError(f"unexpected adapter keys: {unexpected[:5]}")
    device=torch.device(args.device); model.to(device).eval(); vae=AutoencoderKL.from_pretrained(args.vae,torch_dtype=torch.float32).to(device).eval(); scheduler_path=Path(args.irasim_root)/"pretrained_models/scheduler"
    scheduler=(PNDMScheduler.from_pretrained(scheduler_path,beta_start=0.0001,beta_end=0.02,beta_schedule="linear",variance_type="fixed_small") if args.sampler=="pndm" else DPMSolverMultistepScheduler.from_pretrained(scheduler_path)); pipe=Trajectory2VideoGenPipeline(vae=vae,scheduler=scheduler,transformer=model)
    with np.load(args.window,allow_pickle=False) as value: context=value["context_frames"].copy(); target=value["target_frames"].copy(); actions=np.concatenate((value["history_actions"],value["future_actions"])).astype(np.float32)
    action=(actions-np.asarray(adapter["action_mean"]))/np.asarray(adapter["action_std"]); context_tensor=torch.from_numpy(context).to(device).permute(0,3,1,2).float().div(127.5).sub(1)
    generator=torch.Generator(device=device).manual_seed(args.seed)
    with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
        latent=vae.encode(context_tensor).latent_dist.mode().mul(vae.config.scaling_factor).unsqueeze(0)
        if args.initial_prediction:
            if args.sampler != "dpm" or not 0 < args.strength <= 1:
                raise ValueError("diffusion refinement requires --sampler dpm and 0 < --strength <= 1")
            initial_u8=np.load(args.initial_prediction)
            if initial_u8.shape != (8,256,256,3): raise ValueError("initial prediction must have shape [8,256,256,3]")
            initial_tensor=torch.from_numpy(initial_u8.copy()).to(device).permute(0,3,1,2).float().div(127.5).sub(1)
            initial_latent=vae.encode(initial_tensor).latent_dist.mode().mul(vae.config.scaling_factor).unsqueeze(0)
            scheduler.set_timesteps(args.steps,device=device); begin=min(len(scheduler.timesteps)-1,max(0,int(round((1.0-args.strength)*(len(scheduler.timesteps)-1))))); timesteps=scheduler.timesteps[begin:]
            if hasattr(scheduler,"set_begin_index"): scheduler.set_begin_index(begin)
            noise=torch.randn(initial_latent.shape,generator=generator,device=device,dtype=initial_latent.dtype); future=scheduler.add_noise(initial_latent,noise,timesteps[:1]); latents=torch.cat((latent,future),dim=1); action_tensor=torch.from_numpy(action).unsqueeze(0).to(device)
            for timestep in timesteps:
                scaled=scheduler.scale_model_input(latents,timestep); t=timestep.reshape(1).expand(latents.shape[0]); noise_prediction=model(scaled,t,actions=action_tensor,mask_frame_num=5); latents=scheduler.step(noise_prediction,timestep,latents,return_dict=False)[0]; latents=torch.cat((latent,latents[:,5:]),dim=1)
            video=pipe.decode_latents(latents)
        else:
            video,_=pipe(torch.from_numpy(action).unsqueeze(0).to(device),mask_x=latent,video_length=13,height=256,width=256,num_inference_steps=args.steps,guidance_scale=1.0,generator=generator,device=device,output_type="both")
    prediction=video[0,5:].float().add(1).mul(127.5).clamp(0,255).round().byte().permute(0,2,3,1).cpu().numpy(); output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True); np.save(output.with_suffix(".npy"),prediction); mae=np.abs(prediction.astype(np.float32)-target.astype(np.float32)).mean(axis=(1,2,3)); frames=[np.concatenate((prediction[i],target[i]),axis=1) for i in range(8)]; imageio.mimsave(output,frames,fps=3,loop=0)
    pred_tensor=torch.from_numpy(prediction.copy()).permute(0,3,1,2).float().div(255); target_tensor=torch.from_numpy(target.copy()).permute(0,3,1,2).float().div(255); error=(pred_tensor-target_tensor).abs(); dark=target_tensor.mean(1,keepdim=True)<0.28
    report={"format":"track2-irasim-adapter-single-window-v1","window":str(Path(args.window).resolve()),"sampler":args.sampler,"steps":args.steps,"initial_prediction":args.initial_prediction,"strength":args.strength,"seed":args.seed,"metrics":{"rgb_mae_0_255":float(error.mean()*255),"highpass_mae_0_255":float((highpass(pred_tensor)-highpass(target_tensor)).abs().mean()*255),"dark_structure_mae_0_255":float(error.mean(1,keepdim=True)[dark].mean()*255)},"mae_by_frame":mae.tolist(),"output":str(output.resolve())}; output.with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))


if __name__=="__main__": main()
