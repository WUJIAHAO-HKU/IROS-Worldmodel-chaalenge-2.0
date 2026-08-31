#!/usr/bin/env python3
"""One-shot v476 reward-import migration S1 generator; v475 serial inference is frozen."""
from __future__ import annotations

import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import torch

from wam_pipeline.v475_v474_serial_v169_runtime import Track2V475V474SerialV169
from wam_pipeline.v474_v473_median4_parent_runtime import action_feature

PREREG_FORMAT="strict-track2-v476-v475-reward-import-s1-preregistration-v1"
SELECTION_FORMAT="strict-track2-v474-s1-action-only-selection-v1"
RECEIPT_FORMAT="strict-track2-v474-s1-action-only-selection-receipt-v1"
OUTPUT_FORMAT="strict-track2-v476-s1-offline-inputs-v1"

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def treesha(root):
 root=Path(root).resolve();lines=[]
 for p in sorted(x for x in root.rglob("*") if x.is_file()):lines.append(f"{sha(p)}  {p.relative_to(root).as_posix()}\n")
 return hashlib.sha256("".join(lines).encode()).hexdigest()
def json_treesha(root):
 root=Path(root).resolve();items=[]
 for p in sorted(x for x in root.rglob("*") if x.is_file()):items.append([p.relative_to(root).as_posix(),sha(p)])
 return hashlib.sha256(json.dumps(items,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def arrsha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def stable_seed(ep,start,chunk):
 d=hashlib.sha256(f"v474-s1/episode{ep}/start{start}/chunk{chunk}/seed1617".encode()).digest()
 return int.from_bytes(d[:8],"little")%(2**31)
def atomic_json(path,obj):
 path=Path(path);tmp=path.with_name(path.name+".tmp")
 if path.exists() or tmp.exists():raise FileExistsError(path)
 with tmp.open("x",encoding="utf-8") as f:
  json.dump(obj,f,sort_keys=True,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def update(context,history,frames,actions):
 return np.concatenate((context,frames),1)[:,-5:].copy(),np.concatenate((history,actions),1)[:,-4:].copy()
def _source_paths(row):
 raw=row.get("source_windows",row.get("source_window"))
 if isinstance(raw,(str,Path,dict)):raw=[raw]
 if not isinstance(raw,list) or not raw:raise RuntimeError("selection row has no source_windows")
 out=[]
 expected_list=row.get("source_window_sha256")
 if isinstance(expected_list,str):expected_list=[expected_list]
 if expected_list is not None and (not isinstance(expected_list,list) or len(expected_list)!=len(raw)):raise RuntimeError("selection source window SHA cardinality")
 for index,item in enumerate(raw):
  p=Path(item["path"] if isinstance(item,dict) else item).resolve()
  expected=(item.get("sha256") if isinstance(item,dict) else None) or (expected_list[index] if isinstance(expected_list,list) else None)
  if not expected:raise RuntimeError("selection does not freeze source window SHA")
  if expected and sha(p)!=expected:raise RuntimeError(f"window SHA drift: {p}")
  out.append(p)
 return out
def verify_closure(pre,args):
 closure=pre.get("execution_closure")
 if not isinstance(closure,dict):raise RuntimeError("formal prereg missing execution_closure")
 actual={"materializer":Path(closure["materializer"]["path"]),"generator":Path(__file__).resolve(),"auditor":Path(closure["auditor"]["path"]),"s1_wrapper":Path(closure["s1_wrapper"]["path"]),"selection":args.selection.resolve(),"selection_receipt":args.selection_receipt.resolve(),"release_manifest":(args.release/"v475_serial_v169_manifest.json").resolve(),"static_audit":args.static_audit.resolve(),"reward_checkpoint":args.reward_checkpoint.resolve()}
 for key,path in actual.items():
  rec=closure.get(key,{})
  if Path(rec.get("path","")).resolve()!=path or not path.is_file() or sha(path)!=rec.get("sha256"):raise RuntimeError(f"formal closure drift: {key}")
 for key,path,fn in (("release_tree",args.release.resolve(),treesha),("t5_model_tree",args.t5_model.resolve(),json_treesha)):
  rec=closure.get(key,{})
  if Path(rec.get("path","")).resolve()!=path or not path.is_dir() or fn(path)!=rec.get("sha256"):raise RuntimeError(f"formal tree closure drift: {key}")
 if closure["release_manifest"]["sha256"]!="3d505839d10c3d4fbd88bcc6b0e01f2802efb8b2b3047095def2209d84d64a1d" or closure["release_tree"]["sha256"]!="e737d96d07e59d120143811669ea0f5c0cc1855ef14be252dfdf09666b9c32a3" or closure["static_audit"]["sha256"]!="a1fda80ce1aca8ffe33ac1e4e36167df48e3318c33a3ce64a3bac0d70bb037af":raise RuntimeError("v475 immutable release closure drift")
def load_row(row,horizon):
 paths=_source_paths(row);future=[];target=[]
 with np.load(paths[0],allow_pickle=False) as z:
  context=np.asarray(z["context_frames"],np.uint8).copy();history=np.asarray(z["history_actions"],np.float32).copy()
 for p in paths:
  with np.load(p,allow_pickle=False) as z:
   future.append(np.asarray(z["future_actions"],np.float32));target.append(np.asarray(z["target_frames"],np.uint8))
 future=np.concatenate(future)[:horizon].copy();target=np.concatenate(target)[:horizon].copy()
 if context.shape!=(5,256,256,3) or history.shape!=(4,14) or future.shape!=(horizon,14) or target.shape!=(horizon,256,256,3):raise RuntimeError("window array shape drift")
 for key,value in (("history_action_sha256",history),("future32_action_sha256" if horizon==32 else "future8_action_sha256",future),("target_frames_sha256",target)):
  if row.get(key) and arrsha(value)!=row[key]:raise RuntimeError(f"row {key} drift")
 return {"context":context,"history":history,"future":future,"target":target}
def score(model,frames,prompts,device,batch=32):
 n,t=frames.shape[:2];flat=frames.reshape(n*t,*frames.shape[2:]);texts=[p for p in prompts for _ in range(t)];out=[]
 for begin in range(0,len(flat),batch):
  end=min(begin+batch,len(flat));x=torch.from_numpy(np.ascontiguousarray(flat[begin:end])).permute(0,3,1,2).float().div(255).to(device)
  with torch.inference_mode():y=model.compute_reward(x,task_descriptions=texts[begin:end])
  out.extend(y.detach().float().cpu().tolist())
 return np.asarray(out,np.float32).reshape(n,t)
def normalized_rows(selection,key,count):
 rows=selection.get(key)
 if not isinstance(rows,list) or len(rows)!=count:raise RuntimeError(f"selection {key} count")
 return rows
def override_features(selection):
 raw=selection.get("shuffle",{}).get("override_action_features")
 if raw is None:raise RuntimeError("selection missing frozen override_features")
 value=np.asarray(raw,np.float32)
 if value.shape!=(32,4,54) or not np.isfinite(value).all():raise RuntimeError("override feature shape")
 expected=selection["shuffle"]["override_action_features_sha256"]
 if arrsha(value)!=expected:raise RuntimeError("override feature SHA drift")
 return value

def rollout_right(runtime,rows,loaded,override,batch_size):
 # batch_size is a frozen CLI/output-assembly parameter only.  Backend calls follow
 # the contract's exact row-major scalar order: selection row, then chunk 0..3.
 if batch_size!=4:raise RuntimeError("frozen output assembly batch size")
 blocks=[]
 for ri,(row,item) in enumerate(zip(rows,loaded)):
  prompt=str(row["instruction"]);episode=int(row["episode"]);start=int(row["start"]);future=item["future"]
  independent_context=item["context"].copy();independent_history=item["history"].copy();true_context=item["context"].copy();true_history=item["history"].copy()
  chunks={k:[] for k in ("independent_v169","same_request_v169","v475_true","v475_override")};decisions=[];override_decisions=[]
  for ci in range(4):
   actions=future[ci*8:(ci+1)*8];seed=stable_seed(episode,start,ci)
   if "chunk_seeds" in row and int(row["chunk_seeds"][ci])!=seed:raise RuntimeError("selection seed drift")
   independent=runtime.v169.predict(independent_context,independent_history,actions,seed,prompt)
   same,true,over,d,od=runtime.predict_one_s1_pair(true_context,true_history,actions,seed,prompt,override[ri,ci])
   for key in ("gate","explicit_right","postclose","no_transport","phase","context_neighbor_sha256"):
    if d[key]!=od[key]:raise RuntimeError(f"override changed {key}")
   for key,value in (("independent_v169",independent),("same_request_v169",same),("v475_true",true),("v475_override",over)):
    value=np.asarray(value)
    if value.shape!=(8,256,256,3) or value.dtype!=np.uint8:raise RuntimeError(f"{key} shape")
    chunks[key].append(value)
   decisions.append(d);override_decisions.append(od)
   independent_context=np.concatenate((independent_context,independent),axis=0)[-5:].copy();independent_history=np.concatenate((independent_history,actions),axis=0)[-4:].copy()
   true_context=np.concatenate((true_context,true),axis=0)[-5:].copy();true_history=np.concatenate((true_history,actions),axis=0)[-4:].copy()
  blocks.append({**{k:np.concatenate(v,axis=0) for k,v in chunks.items()},"decisions":decisions,"override_decisions":override_decisions})
 return {**{k:np.stack([b[k] for b in blocks]) for k in ("independent_v169","same_request_v169","v475_true","v475_override")},"decisions":[b["decisions"] for b in blocks],"override_decisions":[b["override_decisions"] for b in blocks]}

def rollout_left(runtime,rows,loaded,batch_size):
 out=[];base=[];decisions=[]
 for bb in range(0,12,batch_size):
  rr=rows[bb:bb+batch_size];xx=loaded[bb:bb+batch_size];n=len(rr);context=np.stack([x["context"] for x in xx]);history=np.stack([x["history"] for x in xx]);future=np.stack([x["future"] for x in xx]);seeds=np.asarray([stable_seed(int(r["episode"]),int(r["start"]),0) for r in rr]);prompts=[str(r["instruction"]) for r in rr]
  if any(int(r.get("seed",-1))!=int(seeds[i]) for i,r in enumerate(rr)):raise RuntimeError("left selection seed drift")
  pairs=[runtime.predict_one_with_baseline(context[i],history[i],future[i],int(seeds[i]),prompts[i]) for i in range(n)]
  b=np.stack([p[0] for p in pairs]);o=np.stack([p[1] for p in pairs]);d=[p[2] for p in pairs]
  if not np.array_equal(b,o):raise RuntimeError("left S1 not bitexact")
  base.append(b);out.append(o);decisions.extend(d)
 return np.concatenate(base),np.concatenate(out),decisions

def interface_probe(runtime,right,right_data,left,left_data):
 pairs=[(right[0],right_data[0]),(left[0],left_data[0]),(right[1],right_data[1]),(left[1],left_data[1])]
 context=np.stack([x[1]["context"] for x in pairs]);history=np.stack([x[1]["history"] for x in pairs]);future=np.stack([x[1]["future"][:8] for x in pairs]);seeds=np.asarray([stable_seed(int(r["episode"]),int(r["start"]),0) for r,_ in pairs]);prompts=[str(r["instruction"]) for r,_ in pairs]
 base,batch,_=runtime.predict_batch_with_baseline(context,history,future,seeds,prompts)
 scalar=np.stack([runtime.predict(context[i],history[i],future[i],int(seeds[i]),prompts[i]) for i in range(4)])
 perm=np.asarray([2,0,3,1]);p=runtime.predict_batch(context[perm],history[perm],future[perm],seeds[perm],[prompts[i] for i in perm]);restored=np.empty_like(p);restored[perm]=p
 return {"scalar_batch_bitexact":bool(np.array_equal(scalar,batch)),"batch_permutation_bitexact":bool(np.array_equal(restored,batch)),"mixed_left_bitexact":bool(np.array_equal(batch[[1,3]],base[[1,3]]))}

def main():
 ap=argparse.ArgumentParser()
 for n in ("preregistration","selection","selection-receipt","release","static-audit","reward-checkpoint","t5-model","output","report"):ap.add_argument("--"+n,required=True,type=Path)
 ap.add_argument("--device",default="cuda");ap.add_argument("--inference-batch-size",type=int,default=4);ap.add_argument("--reward-batch-size",type=int,default=32);a=ap.parse_args()
 if a.output.exists() or a.report.exists():raise FileExistsError("v476 S1 output exists")
 if a.inference_batch_size!=4 or a.reward_batch_size!=32:raise RuntimeError("frozen S1 batch sizes")
 pre=json.loads(a.preregistration.read_text());sel=json.loads(a.selection.read_text());rec=json.loads(a.selection_receipt.read_text())
 if pre.get("format")!=PREREG_FORMAT or pre.get("status")!="preregistered_public_s1_one_shot_authorized" or sel.get("format")!=SELECTION_FORMAT or rec.get("format")!=RECEIPT_FORMAT or rec.get("passed") is not True or not all(rec.get("checks",{}).values()):raise RuntimeError("S1 prerequisite invalid")
 auth=pre.get("authorization",{})
 if pre.get("run_count_exact")!=1 or auth.get("s1_run_authorized") is not True or auth.get("zero_update_authorized_before_s1_pass") is not False or auth.get("policy_update_authorized") is not False or auth.get("formal_rl_authorized") is not False:raise RuntimeError("S1 authorization boundary")
 if pre.get("orchestration_boundary")!={"launcher_is_postregistration_orchestrator":True,"launcher_must_bind_formal_preregistration_sha256":True,"launcher_is_not_part_of_execution_closure":True}:raise RuntimeError("S1 orchestration boundary")
 if rec.get("selection_sha256")!=sha(a.selection):raise RuntimeError("selection receipt SHA")
 verify_closure(pre,a)
 release_manifest=a.release/"v475_serial_v169_manifest.json"
 if sha(release_manifest)!="3d505839d10c3d4fbd88bcc6b0e01f2802efb8b2b3047095def2209d84d64a1d":raise RuntimeError("v475 manifest drift")
 right=normalized_rows(sel,"right",32);left=normalized_rows(sel,"left",12);override=override_features(sel)
 right_data=[load_row(r,32) for r in right];left_data=[load_row(r,8) for r in left]
 runtime=Track2V475V474SerialV169(a.release,a.device)
 # The real-v169 composition check is a preflight: no full S1 RGB or reward work may precede it.
 probes=interface_probe(runtime,right,right_data,left,left_data)
 if not all(probes.values()):raise RuntimeError(f"S1 interface probe failed: {probes}")
 generated=rollout_right(runtime,right,right_data,override,a.inference_batch_size);left_base,left_out,left_decisions=rollout_left(runtime,left,left_data,a.inference_batch_size)
 target=np.stack([x["target"] for x in right_data]);mask=np.asarray([[bool(x["gate"]) for x in row] for row in generated["decisions"]],np.bool_)
 expected=np.asarray(sel["postclose"]["mask32x4"],np.bool_)
 if mask.shape!=(32,4) or not np.array_equal(mask,expected):raise RuntimeError("runtime gate differs from action-only mask")
 # Freeze all RGB before the only reward import/load.
 rgb={"independent_v169_frames":generated["independent_v169"],"same_request_v169_frames":generated["same_request_v169"],"v476_true_frames":generated["v475_true"],"v476_action_override_frames":generated["v475_override"],"target_frames":target,"left_same_request_scalar_v169_frames":left_base,"left_v476_frames":left_out}
 rgb_sha={k:arrsha(v) for k,v in rgb.items()}
 from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel
 device=torch.device(a.device);reward=RoboTwinT5CrossAttnRewardModel.from_pretrained(str(a.reward_checkpoint),config={"t5_model_name":str(a.t5_model)}).to(device).eval().requires_grad_(False);prompts=[str(r["instruction"]) for r in right]
 rewards={"independent_v169_reward":score(reward,rgb["independent_v169_frames"],prompts,device,a.reward_batch_size),"v476_true_reward":score(reward,rgb["v476_true_frames"],prompts,device,a.reward_batch_size),"target_reward":score(reward,target,prompts,device,a.reward_batch_size)}
 a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_name(a.output.name+".tmp.npz")
 np.savez_compressed(tmp,format=np.asarray(OUTPUT_FORMAT),**rgb,**rewards,episode=np.asarray([r["episode"] for r in right],np.int64),start=np.asarray([r["start"] for r in right],np.int64),phase=np.asarray([r["phase"] for r in right]),gate=mask,no_transport=np.asarray([[bool(x["no_transport"]) for x in row] for row in generated["decisions"]],np.bool_),left_episode=np.asarray([r["episode"] for r in left],np.int64))
 os.replace(tmp,a.output)
 report={"format":"strict-track2-v476-s1-generation-report-v1","passed":True,"output":str(a.output.resolve()),"output_sha256":sha(a.output),"selection_sha256":sha(a.selection),"preregistration_sha256":sha(a.preregistration),"rgb_array_sha256":rgb_sha,"right_decisions":generated["decisions"],"right_override_decisions":generated["override_decisions"],"left_decisions":left_decisions,"interface_probes":probes,"reward_load_count":1,"guards":{"selection_or_calibration_used_rgb":False,"selection_or_calibration_used_reward":False,"runtime_new_bnt_calls":0,"offline_frozen_bnt_residualization_only":True,"policy_updates":0,"hidden_or_final_data":False}}
 atomic_json(a.report,report);print(json.dumps({"passed":True,"output":str(a.output),"report":str(a.report)},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
