#!/usr/bin/env python3
"""Preregister v462 only after a passed immutable v456 final audit receipt."""
from __future__ import annotations
import argparse,hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def closure_record(base,base_kind,path,declared_sha256=None):
 base=Path(base).resolve();path=Path(path).resolve()
 if path!=base and base not in path.parents:raise RuntimeError(f"v462 closure path escapes {base_kind}: {path}")
 if not path.is_file():raise RuntimeError(f"v462 closure file missing: {path}")
 digest=sha(path)
 if declared_sha256 is not None and digest!=declared_sha256:raise RuntimeError(f"v462 manifest-declared SHA drift: {path}")
 return {"base_kind":base_kind,"relative":str(path.relative_to(base)),"resolved_path":str(path),"sha256":digest}
def tree(root,base_kind):
 root=Path(root).resolve()
 return [closure_record(root,base_kind,path) for path in sorted(root.rglob("*")) if path.is_file()]
def library_closure(release,library):
 release=Path(release).resolve();library=Path(library).resolve();base=release/"v168_release/base_release";manifest_path=base/"release_manifest.json";manifest=json.loads(manifest_path.read_text());records=[]
 for relative,digest in sorted(manifest["sha256"].items()):records.append(closure_record(base,"v169_base_release",base/relative,digest))
 split=manifest["retrieval"]["split_manifest"];records.append(closure_record(library,"v169_library",library/split))
 windows=(library/manifest["retrieval"]["windows_directory"]).resolve()
 if windows!=library and library not in windows.parents:raise RuntimeError("v462 retrieval windows escape v169 library")
 records.extend(closure_record(library,"v169_library",path) for path in sorted(windows.rglob("*")) if path.is_file())
 keys=[(row["base_kind"],row["relative"]) for row in records]
 if len(keys)!=len(set(keys)):raise RuntimeError("v462 duplicate closure records")
 return records
def arrsha(value):
 import numpy as np
 return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()
def request_phase(row,cache):
 import h5py,numpy as np
 source=Path(row["source_hdf5"])
 if source not in cache:
  if sha(source)!=row["source_hdf5_sha256"]:raise RuntimeError("v462 HDF5 SHA drift")
  with h5py.File(source,"r") as h:cache[source]=np.asarray(h["joint_action/vector"],np.float32)
 actions=cache[source];start=int(row["start"]);history=actions[start:start+4];future=actions[start+4:start+12];anchor=history[-1,7:13];delta=future[:,7:13]-anchor;branches={"factual":future.copy()}
 for name,scale in (("no_transport",0.),("scale_0p4",.4),("scale_1p25",1.25),("reverse_direction_0p4",-.4)):
  value=future.copy();value[:,7:13]=anchor+scale*delta;branches[name]=value
 branches["factual_duplicate"]=future.copy();endpoint={n:float(np.linalg.norm(v[-1,7:13]-future[-1,7:13])) for n,v in branches.items() if n not in ("factual","factual_duplicate")};fullpath={n:float(np.linalg.norm(v[:,7:13]-future[:,7:13])) for n,v in branches.items() if n not in ("factual","factual_duplicate")}
 if arrsha(history)!=row["history_action_sha256"] or {n:arrsha(v) for n,v in branches.items()}!=row["branch_action_sha256"] or endpoint!=row["endpoint_diffs"] or fullpath!=row["path_diffs"] or min(endpoint.values())<.01 or min(fullpath.values())<.01:raise RuntimeError("v462 action closure drift")
 q0=history[-1,7:13];path=np.concatenate((q0[None],future[:,7:13]),0);step=np.linalg.norm(np.diff(path,axis=0),axis=1);total=float(step.sum());score=.5 if total<=1e-8 else float(step[:4].sum()/total);tie=hashlib.sha256(np.ascontiguousarray(np.concatenate((history,future),0),np.float32).view(np.uint8)).hexdigest();return score,tie
def shuffle_map(rows,cache):
 groups={}
 for row in rows:groups.setdefault(int(row["episode"]),[]).append(row)
 result={}
 for episode,group in sorted(groups.items()):
  evidence={id(row):request_phase(row,cache) for row in group};ordered=sorted(group,key=lambda row:evidence[id(row)]);n=len(ordered)
  if n<2:raise RuntimeError(f"v462 phase shuffle requires >=2 contexts episode={episode}")
  offset=max(1,n//2)
  for rank,row in enumerate(ordered):
   donor=ordered[(rank+offset)%n];result[f"{episode}:{int(row['start'])}"]={"donor_episode":episode,"donor_start":int(donor["start"]),"phase_score":evidence[id(row)][0],"donor_phase_score":evidence[id(donor)][0],"offset":offset,"group_size":n}
 return result
def main():
 p=argparse.ArgumentParser()
 for n in ("v456-final-receipt","v456-selection","v456-generation-report","v456-dataset","v169-release","v169-library","v169-runtime-source","trainer","runtime","output"):p.add_argument(f"--{n}",type=Path,required=True)
 a=p.parse_args();receipt=json.loads(a.v456_final_receipt.read_text())
 # This is deliberately the first dependency read: no v456 rows are visible before final PASS.
 if receipt.get("format")!="strict-track2-v461-endpoint200-batch-audit-v1" or receipt.get("mode")!="final" or receipt.get("passed") is not True or not all(receipt.get("checks",{}).values()) or receipt.get("guards",{}).get("rl_authorized") is not False:raise RuntimeError("v462 blocked: v461 immutable final receipt has not passed")
 selection=json.loads(a.v456_selection.read_text());generation=json.loads(a.v456_generation_report.read_text())
 if selection.get("format")!="strict-track2-v461-endpoint200-preregistration-v1" or generation.get("format")!="strict-track2-v461-endpoint200-generation-report-v1" or generation.get("passed") is not True or generation.get("exact_contexts")!=200 or generation.get("exact_batches")!=10:raise RuntimeError("v462 v461 selection/generation contract failed")
 contexts=selection["contexts"];episodes=sorted({int(x["episode"]) for x in contexts});folds=[]
 for fold in range(5):
  held=sorted({int(x["episode"]) for x in contexts if int(x["fold"])==fold});fit=sorted(set(episodes)-set(held));folds.append({"fold":fold,"holdout_episodes":held,"fit_episodes":fit})
 if len(contexts)!=200 or len(episodes)!=15 or any(len(x["holdout_episodes"])!=3 or len(x["fit_episodes"])!=12 for x in folds):raise RuntimeError("v462 episode-grouped 5fold contract failed")
 files=[]
 for batch in range(10):
  root=a.v456_dataset/f"batch_{batch:03d}";report=root/"batch_report.json";files.append({"relative":str(report.relative_to(a.v456_dataset)),"sha256":sha(report)})
  for row in sorted((root/"rows").iterdir()):
   for name in ("endpoint.npz","receipt.json"):
    path=row/name;files.append({"relative":str(path.relative_to(a.v456_dataset)),"sha256":sha(path)})
 if len(files)!=410:raise RuntimeError(f"v462 expected 410 frozen v456 files, got {len(files)}")
 v169_manifest=a.v169_release/"v169_arm_routed_manifest.json";library_manifest=a.v169_library/"release_manifest.json"
 if not library_manifest.is_file():library_manifest=a.v169_library/"splits/adjust_bottle_50episodes_full.json"
 cache={};mappings=[]
 for fold in range(5):
  fit_rows=[x for x in contexts if int(x["fold"])!=fold];hold_rows=[x for x in contexts if int(x["fold"])==fold];mappings.append({"fold":fold,"fit_episodes":sorted({int(x["episode"]) for x in fit_rows}),"holdout_episodes":sorted({int(x["episode"]) for x in hold_rows}),"fit":shuffle_map(fit_rows,cache),"holdout":shuffle_map(hold_rows,cache)})
 ordered=[]
 for batch in range(10):ordered.extend(sorted((x for x in contexts if int(x["batch_id"])==batch),key=lambda x:f"episode{int(x['episode'])}_start{int(x['start']):05d}"))
 schedules=[]
 for fold in range(5):
  fit_contexts=[i for i,x in enumerate(ordered) if int(x["fold"])!=fold];fit_ids=np.asarray([5*i+b for i in fit_contexts for b in range(5)]);rng=np.random.default_rng(1613+fold);schedule=[{"sample_ids":rng.choice(fit_ids,size=8,replace=False).tolist(),"context_id":fit_contexts[step%len(fit_contexts)]} for step in range(50)];encoded=json.dumps(schedule,sort_keys=True,separators=(",",":"));schedules.append({"fold":fold,"sha256":hashlib.sha256(encoded.encode()).hexdigest(),"same_for_heads":["action","context_only","phase_shuffle"],"optimizer_independent":True})
 rng=np.random.default_rng(1713);all200_schedule=[{"sample_ids":rng.choice(np.arange(1000),size=8,replace=False).tolist(),"context_id":step%200} for step in range(50)];all200_schedule_sha=hashlib.sha256(json.dumps(all200_schedule,sort_keys=True,separators=(",",":")).encode()).hexdigest()
 payload={"format":"strict-track2-v462-endpoint-residual-5fold-preregistration-v2","created_at":datetime.now(timezone.utc).isoformat(),"classification":"public-train endpoint world-model parent pilot only; no reward, policy training, RL, dev/final, or submission","seed":1613,"v461":{"final_receipt":{"path":str(a.v456_final_receipt.resolve()),"sha256":sha(a.v456_final_receipt)},"selection":{"path":str(a.v456_selection.resolve()),"sha256":sha(a.v456_selection)},"generation_report":{"path":str(a.v456_generation_report.resolve()),"sha256":sha(a.v456_generation_report)},"dataset":str(a.v456_dataset.resolve()),"files":files,"contexts":200,"endpoint_samples":1000,"branches":["factual","no_transport","scale_0p4","scale_1p25","reverse_direction_0p4"]},"folds":folds,"phase_shuffle":{"formula":"request-only q0/history+future right6; score=.5 if path<=1e-8 else first4_step_sum/total; within episode sort(score,sha256(float32 history||factual future)); offset=max(1,n//2)","mappings":mappings},"v169":{"release":str(a.v169_release.resolve()),"library":str(a.v169_library.resolve()),"base_release":str((a.v169_release/"v168_release/base_release").resolve()),"runtime_source":{"path":str(a.v169_runtime_source.resolve()),"sha256":sha(a.v169_runtime_source)},"release_manifest_sha256":sha(v169_manifest),"library_manifest":str(library_manifest.resolve()),"library_manifest_sha256":sha(library_manifest),"release_files":tree(a.v169_release,"v169_release"),"library_files":library_closure(a.v169_release,a.v169_library)},"source":{"trainer_path":str(a.trainer.resolve()),"trainer_sha256":sha(a.trainer),"runtime_path":str(a.runtime.resolve()),"runtime_sha256":sha(a.runtime)},"training":{"heads_per_fold":["action","context_only","phase_shuffle"],"steps_per_head":50,"batch_size":8,"working_resolution":128,"channels":16,"optimizer":"AdamW","learning_rate":0.0003,"weight_decay":0.0001,"gradient_clip":1.0,"endpoint_l1":1.0,"gradient_l1":0.25,"same_context_action_delta_l1":0.5,"precision":"bf16","all200_action_head_steps_after_oof_pass":50},"features":{"normalized_actions":168,"right6_endpoint_delta":6,"right6_anchor_relative_path":48,"request_only_postclose":1,"arm_tokens":2,"postclose_formula":"shape/finite and history[-1,13]<.5 and all future[:,13]<.5"},"kill_gate":{"continuous_over_v169_max":0.95,"uint8_over_v169_max":1.0,"factual_branch_max":1.0,"each_counterfactual_branch_max":0.98,"action_over_context_only_max":0.95,"action_over_phase_shuffle_max":0.97,"positive_target_delta_cosine_fraction_min":0.60,"distinct_action_prediction_nonzero_fraction_min":0.95,"passing_folds_min":4,"improved_episodes_min":12},"runtime":{"base":"frozen v169","right_postclose_frame8_only":True,"left_g0_frames0_to6_bitexact":True,"four_apis":["predict","predict_with_baseline","predict_batch","predict_batch_with_baseline"],"official_reward_modified":False,"policy_modified":False},"guards":{"v461_final_pass_required":True,"development_or_final_used":False,"reward_loaded":False,"policy_updates":0,"rl_authorized":False,"real_submission":False}}
 payload["common_v169_seed"]="sha256('v462/1613/{episode}/{start}')[:8] little-endian mod 2**31; branch-independent";payload["training_schedules"]=schedules;payload["all200_schedule_sha256"]=all200_schedule_sha;payload["context_only_semantics"]="same branch-specific frozen-v169 visual baseline retained; only explicit normalized/action endpoint/path features zeroed; arm tokens and postclose=1 retained";payload["phase_shuffle_semantics"]="same branch-specific frozen-v169 visual baseline retained; explicit action features replaced by fixed within-episode donor"
 if a.output.exists() or a.output.with_name(a.output.name+".tmp").exists():raise RuntimeError("refusing to overwrite v462 preregistration/tmp")
 a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_name(a.output.name+".tmp")
 with tmp.open("w") as f:json.dump(payload,f,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
 os.replace(tmp,a.output);descriptor=os.open(a.output.parent,os.O_RDONLY);os.fsync(descriptor);os.close(descriptor);print(a.output);return 0
if __name__=="__main__":raise SystemExit(main())
