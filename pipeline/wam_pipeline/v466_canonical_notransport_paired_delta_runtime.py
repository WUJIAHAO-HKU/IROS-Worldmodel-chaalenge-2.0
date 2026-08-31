"""V466 frozen-v169 canonical-no-transport appearance + paired-dynamics parent."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from .v169_arm_routed_runtime import Track2V169ArmRoutedRuntime

FORMAT="track2-v466-canonical-notransport-paired-delta-parent-release-v1"
CHECKPOINT_FORMAT="strict-track2-v466-canonical-notransport-paired-delta-checkpoint-v1"
WORKING_RESOLUTION=128;ACTION_FEATURE_DIM=223
FEATURE_SCHEMA={"dim":223,"normalized_actions":168,"right6_endpoint_delta":6,"right6_anchor_relative_path":48,"postclose":1}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(8<<20),b""):h.update(block)
 return h.hexdigest()
def explicit_right(text):
 value=str(text or "").lower();return "right arm" in value and "left arm" not in value
def endpoint_gate(history,future,instruction):
 h=np.asarray(history);f=np.asarray(future);ok=h.shape==(4,14) and f.shape==(8,14) and np.isfinite(h).all() and np.isfinite(f).all() and explicit_right(instruction) and float(h[-1,13])<.5 and bool(np.all(f[:,13]<.5))
 return {"gate":bool(ok),"phase":"postclose" if ok else "g0"}
def instruction_tokens(instructions):return np.asarray([(float(explicit_right(x)),float("left arm" in str(x or "").lower())) for x in instructions],np.float32)
def canonical_no_transport(history,future):
 h=np.asarray(history,np.float32);f=np.asarray(future,np.float32).copy()
 if h.ndim==2:h=h[None];f=f[None]
 f[:,:,7:13]=h[:,-1,None,7:13]
 return f
def no_transport_mask(history,future):
 h=np.asarray(history,np.float32);f=np.asarray(future,np.float32)
 if h.ndim==2:h=h[None];f=f[None]
 return np.asarray([np.array_equal(row[:,7:13],np.broadcast_to(anchor[7:13],(8,6))) for anchor,row in zip(h[:,-1],f)],bool)
def action_features(history,future,mean,std):
 h=np.asarray(history,np.float32);f=np.asarray(future,np.float32);m=np.asarray(mean,np.float32);s=np.asarray(std,np.float32)
 if h.ndim==2:h=h[None];f=f[None]
 actions=np.concatenate((h,f),1);normalized=(actions-m)/s;anchor=h[:,-1,7:13];endpoint=f[:,-1,7:13]-anchor;path=(f[:,:,7:13]-anchor[:,None]).reshape(len(h),48);postclose=((h[:,-1,13]<.5)&np.all(f[:,:,13]<.5,axis=1)).astype(np.float32)[:,None]
 return np.concatenate((normalized.reshape(len(h),-1),endpoint,path,postclose),1).astype(np.float32)
def appearance_features(n):
 value=np.zeros((int(n),ACTION_FEATURE_DIM),np.float32);value[:,-1]=1.;return value

class EndpointResidualUNet128FiLM(nn.Module):
 def __init__(self,channels=16):
  super().__init__();self.channels=int(channels);self.enc1=nn.Sequential(nn.Conv2d(6,channels,3,padding=1),nn.SiLU(),nn.Conv2d(channels,channels,3,padding=1),nn.SiLU());self.enc2=nn.Sequential(nn.Conv2d(channels,2*channels,3,stride=2,padding=1),nn.SiLU(),nn.Conv2d(2*channels,2*channels,3,padding=1),nn.SiLU());self.condition=nn.Sequential(nn.Linear(ACTION_FEATURE_DIM+2,128),nn.SiLU(),nn.Linear(128,4*channels));self.up=nn.ConvTranspose2d(2*channels,channels,4,stride=2,padding=1);self.decode=nn.Sequential(nn.Conv2d(2*channels,channels,3,padding=1),nn.SiLU(),nn.Conv2d(channels,channels,3,padding=1),nn.SiLU());self.output=nn.Conv2d(channels,3,3,padding=1);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
 def forward(self,baseline_endpoint,context_last,condition_features,arm_tokens):
  b,_,h,w=baseline_endpoint.shape
  if context_last.shape!=(b,3,h,w) or condition_features.shape!=(b,ACTION_FEATURE_DIM) or arm_tokens.shape!=(b,2):raise ValueError("v466 tensor shape contract")
  base=F.interpolate(baseline_endpoint,(128,128),mode="bilinear",align_corners=False);context=F.interpolate(context_last,(128,128),mode="bilinear",align_corners=False);skip=self.enc1(torch.cat((base,context),1));encoded=self.enc2(skip);film=self.condition(torch.cat((condition_features,arm_tokens),1)).reshape(b,2,2*self.channels);gamma=.1*torch.tanh(film[:,0])[:,:,None,None];bias=film[:,1,:,None,None];decoded=self.up(encoded*(1+gamma)+bias);decoded=self.decode(torch.cat((decoded,skip),1));residual=255.*torch.tanh(self.output(decoded));return F.interpolate(residual,(h,w),mode="bilinear",align_corners=False)

class Track2V466CanonicalNoTransportPairedDeltaParent:
 def __init__(self,release_dir,device="cuda"):
  root=Path(release_dir).resolve();manifest=json.loads((root/"v466_canonical_notransport_paired_delta_manifest.json").read_text());appearance=(root/manifest["appearance_checkpoint"]).resolve();dynamics=(root/manifest["dynamics_checkpoint"]).resolve();v169_release=Path(manifest["canonical_v169_release"]).resolve();v169_library=Path(manifest["canonical_v169_library"]).resolve()
  if manifest.get("format")!=FORMAT or manifest.get("official_reward_runtime_used") is not False or root not in appearance.parents or root not in dynamics.parents or sha(Path(__file__))!=manifest["sha256"]["runtime_source"]:raise RuntimeError("bad v466 release")
  if sha(appearance)!=manifest["sha256"]["appearance_checkpoint"] or sha(dynamics)!=manifest["sha256"]["dynamics_checkpoint"]:raise RuntimeError("v466 checkpoint hash drift")
  ast=torch.load(appearance,map_location="cpu",weights_only=False);dst=torch.load(dynamics,map_location="cpu",weights_only=False)
  for state,scope in ((ast,"all200-appearance"),(dst,"all200-dynamics-action")):
   if state.get("format")!=CHECKPOINT_FORMAT or state.get("training_scope")!=scope or state.get("step")!=50 or state.get("precision")!="bf16" or state.get("feature_schema")!=FEATURE_SCHEMA or state.get("runtime_sha256")!=manifest["sha256"]["runtime_source"] or state.get("closure_digest")!=manifest["closure_digest"]:raise RuntimeError("bad v466 checkpoint contract")
  self.device=torch.device(device);self.appearance=EndpointResidualUNet128FiLM(ast["channels"]).to(self.device);self.dynamics=EndpointResidualUNet128FiLM(dst["channels"]).to(self.device);self.appearance.load_state_dict(ast["model"]);self.dynamics.load_state_dict(dst["model"]);self.appearance.eval();self.dynamics.eval();self.mean=np.asarray(dst["action_mean"],np.float32);self.std=np.asarray(dst["action_std"],np.float32)
  if self.mean.shape!=(14,) or self.std.shape!=(14,) or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std<=0):raise RuntimeError("bad v466 normalization")
  for key,path in (("v169_release_manifest",v169_release/"v169_arm_routed_manifest.json"),("v169_library_manifest",Path(manifest["v169_library_manifest"]))):
   if not path.is_file() or sha(path)!=manifest["sha256"][key]:raise RuntimeError(f"bad {key}")
  self.v169=Track2V169ArmRoutedRuntime(v169_release,v169_library,device)
 def _v169_microbatch(self,context,history,future,seeds,instructions):
  values=[]
  for begin in range(0,len(context),4):values.append(self.v169.predict_batch(context[begin:begin+4],history[begin:begin+4],future[begin:begin+4],seeds[begin:begin+4],instructions[begin:begin+4]))
  return np.concatenate(values) if values else np.empty((0,8,256,256,3),np.uint8)
 @torch.inference_mode()
 def predict_batch_with_baseline(self,context,history,future,seeds,instructions):
  context=np.asarray(context);history=np.asarray(history,np.float32);future=np.asarray(future,np.float32);seeds=np.asarray(seeds);n=len(context)
  if context.ndim!=5 or history.shape!=(n,4,14) or future.shape!=(n,8,14) or seeds.shape!=(n,) or len(instructions)!=n or not np.isfinite(history).all() or not np.isfinite(future).all():raise ValueError("v466 request batch contract")
  original=self._v169_microbatch(context,history,future,seeds,list(instructions))
  if original.shape[:2]!=(n,8) or original.dtype!=np.uint8:raise RuntimeError("v169 original output contract")
  output=original.copy();decisions=[endpoint_gate(h,f,t) for h,f,t in zip(history,future,instructions)];enabled=np.flatnonzero([x["gate"] for x in decisions])
  if len(enabled):
   canonical=canonical_no_transport(history[enabled],future[enabled]);b0_full=self._v169_microbatch(context[enabled],history[enabled],canonical,seeds[enabled],[instructions[i] for i in enabled]);b0=b0_full[:,7];base=torch.as_tensor(b0,dtype=torch.float32,device=self.device).permute(0,3,1,2)/255.;last=torch.as_tensor(context[enabled,-1],dtype=torch.float32,device=self.device).permute(0,3,1,2)/255.;tokens=torch.as_tensor(instruction_tokens([instructions[i] for i in enabled]),device=self.device);afeat=torch.as_tensor(appearance_features(len(enabled)),device=self.device);dfeat=torch.as_tensor(action_features(history[enabled],future[enabled],self.mean,self.std),device=self.device)
   with torch.autocast(device_type=self.device.type,dtype=torch.bfloat16,enabled=self.device.type=="cuda"):
    appearance=self.appearance(base,last,afeat,tokens);dynamics=self.dynamics(base,last,dfeat,tokens)
   mask=no_transport_mask(history[enabled],future[enabled]);dynamics[torch.as_tensor(mask,device=self.device)]=0;candidate=(torch.as_tensor(b0,dtype=torch.float32,device=self.device).permute(0,3,1,2)+appearance+dynamics).clamp(0,255);candidate=np.clip(np.rint(candidate.permute(0,2,3,1).float().cpu().numpy()),0,255).astype(np.uint8)
   for local,index in enumerate(enabled):output[index,7]=candidate[local]
  return original,output,decisions
 def predict_batch(self,*args):return self.predict_batch_with_baseline(*args)[1]
 def predict_with_baseline(self,context,history,future,seed,instruction):
  baseline,output,decision=self.predict_batch_with_baseline(np.asarray(context)[None],np.asarray(history)[None],np.asarray(future)[None],np.asarray([seed]),[instruction]);return baseline[0],output[0],decision[0]
 def predict(self,context,history,future,seed,instruction):return self.predict_with_baseline(context,history,future,seed,instruction)[1]
