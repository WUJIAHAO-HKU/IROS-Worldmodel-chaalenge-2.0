"""V464 frozen-v169 right/post-close endpoint-only residual parent runtime."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from .v169_arm_routed_runtime import Track2V169ArmRoutedRuntime

FORMAT="track2-v464-v169-endpoint-residual-parent-release-v1"
CHECKPOINT_FORMAT="strict-track2-v464-endpoint-residual-checkpoint-v1"
WORKING_RESOLUTION=128
ACTION_FEATURE_DIM=12*14+6+48+1
def sha(path):
 h=hashlib.sha256();h.update(Path(path).read_bytes());return h.hexdigest()
def explicit_right(text):
 value=str(text or "").lower();return "right arm" in value and "left arm" not in value
def endpoint_gate(history,future,instruction):
 h=np.asarray(history);f=np.asarray(future);ok=h.shape==(4,14) and f.shape==(8,14) and np.isfinite(h).all() and np.isfinite(f).all() and explicit_right(instruction) and float(h[-1,13])<.5 and bool(np.all(f[:,13]<.5))
 return {"gate":bool(ok),"phase":"postclose" if ok else "g0"}
def instruction_tokens(instructions):return np.asarray([(float(explicit_right(x)),float("left arm" in str(x or "").lower())) for x in instructions],np.float32)
def action_features(history,future,mean,std):
 h=np.asarray(history,np.float32);f=np.asarray(future,np.float32);m=np.asarray(mean,np.float32);s=np.asarray(std,np.float32)
 if h.ndim==2:h=h[None];f=f[None]
 actions=np.concatenate((h,f),1);normalized=(actions-m)/s;anchor=h[:,-1,7:13];endpoint=f[:,-1,7:13]-anchor;path=(f[:,:,7:13]-anchor[:,None]).reshape(len(h),48);postclose=((h[:,-1,13]<.5)&np.all(f[:,:,13]<.5,axis=1)).astype(np.float32)[:,None]
 return np.concatenate((normalized.reshape(len(h),-1),endpoint,path,postclose),1).astype(np.float32)

class EndpointResidualUNet128FiLM(nn.Module):
 def __init__(self,channels=16):
  super().__init__();self.channels=int(channels)
  self.enc1=nn.Sequential(nn.Conv2d(6,channels,3,padding=1),nn.SiLU(),nn.Conv2d(channels,channels,3,padding=1),nn.SiLU())
  self.enc2=nn.Sequential(nn.Conv2d(channels,2*channels,3,stride=2,padding=1),nn.SiLU(),nn.Conv2d(2*channels,2*channels,3,padding=1),nn.SiLU())
  self.condition=nn.Sequential(nn.Linear(ACTION_FEATURE_DIM+2,128),nn.SiLU(),nn.Linear(128,4*channels))
  self.up=nn.ConvTranspose2d(2*channels,channels,4,stride=2,padding=1);self.decode=nn.Sequential(nn.Conv2d(2*channels,channels,3,padding=1),nn.SiLU(),nn.Conv2d(channels,channels,3,padding=1),nn.SiLU());self.output=nn.Conv2d(channels,3,3,padding=1)
  nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
 def forward(self,baseline_endpoint,context_last,condition_features,arm_tokens):
  b,_,h,w=baseline_endpoint.shape
  if context_last.shape!=(b,3,h,w) or condition_features.shape!=(b,ACTION_FEATURE_DIM) or arm_tokens.shape!=(b,2):raise ValueError("v464 tensor shape contract")
  base=F.interpolate(baseline_endpoint,(128,128),mode="bilinear",align_corners=False);context=F.interpolate(context_last,(128,128),mode="bilinear",align_corners=False);skip=self.enc1(torch.cat((base,context),1));encoded=self.enc2(skip)
  film=self.condition(torch.cat((condition_features,arm_tokens),1)).reshape(b,2,2*self.channels);gamma=.1*torch.tanh(film[:,0])[:,:,None,None];bias=film[:,1,:,None,None];decoded=self.up(encoded*(1+gamma)+bias);decoded=self.decode(torch.cat((decoded,skip),1));residual128=255.*torch.tanh(self.output(decoded));return F.interpolate(residual128,(h,w),mode="bilinear",align_corners=False)

def apply_endpoint(baseline,residual):
 out=np.asarray(baseline).copy();value=np.clip(np.rint(out[7].astype(np.float32)+np.asarray(residual,np.float32)),0,255).astype(np.uint8);out[7]=value;return out

class Track2V464V169EndpointResidualParent:
 def __init__(self,release_dir,device="cuda"):
  root=Path(release_dir).resolve();manifest=json.loads((root/"v464_endpoint_residual_manifest.json").read_text());checkpoint=(root/manifest["checkpoint"]).resolve();v169_release=(root/manifest["v169_release"]).resolve();v169_library=(root/manifest["v169_library"]).resolve()
  if root not in checkpoint.parents or manifest.get("format")!=FORMAT or manifest.get("official_reward_runtime_used") is not False or sha(checkpoint)!=manifest["sha256"]["checkpoint"] or sha(Path(__file__))!=manifest["sha256"]["runtime_source"]:raise RuntimeError("bad v464 release")
  state=torch.load(checkpoint,map_location="cpu",weights_only=False)
  expected_schema={"dim":ACTION_FEATURE_DIM,"normalized_actions":168,"right6_endpoint_delta":6,"right6_anchor_relative_path":48,"postclose":1}
  if state.get("format")!=CHECKPOINT_FORMAT or state.get("training_scope")!="all200" or state.get("precision")!="bf16" or state.get("feature_schema")!=expected_schema or state.get("runtime_sha256")!=manifest["sha256"]["runtime_source"] or state.get("closure_digest")!=manifest.get("closure_digest"):raise RuntimeError("bad v464 checkpoint")
  if root not in v169_release.parents and str(v169_release)!=manifest.get("canonical_v169_release"):raise RuntimeError("v169 release path escape")
  if root not in v169_library.parents and str(v169_library)!=manifest.get("canonical_v169_library"):raise RuntimeError("v169 library path escape")
  self.device=torch.device(device);self.model=EndpointResidualUNet128FiLM(state["channels"]).to(self.device);self.model.load_state_dict(state["model"]);self.model.eval();self.mean=np.asarray(state["action_mean"],np.float32);self.std=np.asarray(state["action_std"],np.float32)
  if self.mean.shape!=(14,) or self.std.shape!=(14,) or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std<=0):raise RuntimeError("bad v464 normalization")
  for key,path in (("v169_release_manifest",v169_release/"v169_arm_routed_manifest.json"),("v169_library_manifest",v169_library/manifest["v169_library_manifest"])):
   if not path.is_file() or sha(path)!=manifest["sha256"][key]:raise RuntimeError(f"bad {key}")
  self.v169=Track2V169ArmRoutedRuntime(v169_release,v169_library,device)
 @torch.inference_mode()
 def predict_batch_with_baseline(self,context,history,future,seeds,instructions):
  context=np.asarray(context);history=np.asarray(history);future=np.asarray(future);n=len(context)
  if context.ndim!=5 or history.shape!=(n,4,14) or future.shape!=(n,8,14) or len(seeds)!=n or len(instructions)!=n or not np.isfinite(history).all() or not np.isfinite(future).all():raise ValueError("v464 request batch contract")
  baseline=self.v169.predict_batch(context,history,future,seeds,instructions)
  if baseline.ndim!=5 or baseline.shape[0]!=n or baseline.shape[1]!=8 or baseline.shape[-1]!=3 or baseline.dtype!=np.uint8:raise RuntimeError("v169 baseline output contract")
  output=baseline.copy();decisions=[endpoint_gate(h,f,t) for h,f,t in zip(history,future,instructions)];enabled=np.asarray([x["gate"] for x in decisions])
  if enabled.any():
   features=torch.as_tensor(action_features(history[enabled],future[enabled],self.mean,self.std),device=self.device);base=torch.as_tensor(baseline[enabled,7],dtype=torch.float32,device=self.device).permute(0,3,1,2)/255.;last=torch.as_tensor(context[enabled,-1],dtype=torch.float32,device=self.device).permute(0,3,1,2)/255.;tokens=torch.as_tensor(instruction_tokens([instructions[i] for i in np.flatnonzero(enabled)]),device=self.device)
   with torch.autocast(device_type=self.device.type,dtype=torch.bfloat16,enabled=self.device.type=="cuda"):res=self.model(base,last,features,tokens)
   res=res.permute(0,2,3,1).float().cpu().numpy()
   for local,index in enumerate(np.flatnonzero(enabled)):output[index]=apply_endpoint(baseline[index],res[local])
  return baseline,output,decisions
 def predict_batch(self,*args):return self.predict_batch_with_baseline(*args)[1]
 def predict_with_baseline(self,context,history,future,seed,instruction):
  base,out,decision=self.predict_batch_with_baseline(np.asarray(context)[None],np.asarray(history)[None],np.asarray(future)[None],np.asarray([seed]),[instruction]);return base[0],out[0],decision[0]
 def predict(self,context,history,future,seed,instruction):return self.predict_with_baseline(context,history,future,seed,instruction)[1]
