#!/usr/bin/env python3
"""Preregister v467 only after a passed immutable v456 final audit receipt."""
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
def inside(path,root):
 path=Path(path).resolve();root=Path(root).resolve();return path==root or root in path.parents
def directory_target_sha(path):
 path=Path(path).resolve();items=[]
 for parent,dirs,files in os.walk(path,followlinks=False):
  for name in sorted(dirs+files):
   item=Path(parent)/name;relative=str(item.relative_to(path))
   if item.is_symlink():items.append(["link",relative,os.readlink(item)])
   elif item.is_file():items.append(["file",relative,sha(item)])
   else:items.append(["dir",relative,None])
 return hashlib.sha256(json.dumps(items,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def file_record(base,base_kind,path,artifact_root,relative=None,declared_sha256=None):
 base=Path(base).absolute();lexical=(base/(relative if relative is not None else Path(path).absolute().relative_to(base))).absolute();resolved=lexical.resolve();artifact_root=Path(artifact_root).resolve()
 if os.path.commonpath((str(base),str(lexical)))!=str(base) or not inside(resolved,artifact_root) or not resolved.is_file():raise RuntimeError(f"v467 closure path escape/missing: {lexical} -> {resolved}")
 digest=sha(resolved)
 if declared_sha256 is not None and digest!=declared_sha256:raise RuntimeError(f"v467 manifest-declared SHA drift: {lexical}")
 return {"record_type":"file","base_kind":base_kind,"relative":str(lexical.relative_to(base)),"lexical_path":str(lexical),"link_target":os.readlink(lexical) if lexical.is_symlink() else None,"resolved_path":str(resolved),"target_sha":digest}
def release_closure(release,artifact_root):
 release=Path(release).absolute();artifact_root=Path(artifact_root).resolve();records=[];seen_files=set();seen_dirs=set();active=set()
 def visit(logical,relative):
  resolved=logical.resolve()
  if not inside(resolved,artifact_root) or not resolved.is_dir():raise RuntimeError(f"v467 release directory escape/missing: {logical}")
  key=(resolved.stat().st_dev,resolved.stat().st_ino)
  if key in active:raise RuntimeError(f"v467 release symlink cycle: {logical}")
  active.add(key);seen_dirs.add(key)
  for child in sorted(logical.iterdir(),key=lambda x:x.name):
   rel=relative/child.name;target=child.resolve()
   if not inside(target,artifact_root):raise RuntimeError(f"v467 release target escapes joint artifact root: {child} -> {target}")
   if child.is_symlink():
    target_sha=directory_target_sha(target) if target.is_dir() else sha(target)
    records.append({"record_type":"directory_symlink" if target.is_dir() else "file_symlink","base_kind":"v169_release_link","relative":str(rel),"lexical_path":str(child.absolute()),"link_target":os.readlink(child),"resolved_path":str(target),"target_sha":target_sha})
   if target.is_dir():
    target_key=(target.stat().st_dev,target.stat().st_ino)
    if target_key in active:raise RuntimeError(f"v467 release symlink cycle: {child}")
    if target_key in seen_dirs:raise RuntimeError(f"v467 duplicate release directory target: {child} -> {target}")
    visit(child,rel)
   elif target.is_file():
    target_key=(target.stat().st_dev,target.stat().st_ino)
    if target_key in seen_files:raise RuntimeError(f"v467 duplicate release file target: {child} -> {target}")
    seen_files.add(target_key)
    if not child.is_symlink():records.append(file_record(release,"v169_release_target" if not inside(target,release.resolve()) else "v169_release",child,artifact_root,rel))
  active.remove(key)
 visit(release,Path("."));keys=[(row["record_type"],row["relative"],row["resolved_path"]) for row in records]
 if len(keys)!=len(set(keys)):raise RuntimeError("v467 duplicate release closure records")
 return sorted(records,key=lambda row:(row["relative"],row["record_type"]))
def closure_record(base,base_kind,path,artifact_root,declared_sha256=None):return file_record(base,base_kind,path,artifact_root,None,declared_sha256)
def library_closure(release,library):
 release=Path(release).absolute();library=Path(library).resolve();base=(release/"v168_release/base_release").resolve();manifest_path=base/"release_manifest.json";manifest=json.loads(manifest_path.read_text());records=[]
 for relative,digest in sorted(manifest["sha256"].items()):records.append(closure_record(base,"v169_base_release",base/relative,library,digest))
 split=manifest["retrieval"]["split_manifest"];records.append(closure_record(library,"v169_library",library/split,library))
 windows=(library/manifest["retrieval"]["windows_directory"]).resolve()
 if windows!=library and library not in windows.parents:raise RuntimeError("v467 retrieval windows escape v169 library")
 records.extend(closure_record(library,"v169_library",path,library) for path in sorted(windows.rglob("*")) if path.is_file())
 keys=[(row["base_kind"],row["relative"],row["resolved_path"]) for row in records]
 if len(keys)!=len(set(keys)):raise RuntimeError("v467 duplicate closure records")
 return records
def arrsha(value):
 import numpy as np
 return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()
def request_phase(row,cache):
 import h5py,numpy as np
 source=Path(row["source_hdf5"])
 if source not in cache:
  if sha(source)!=row["source_hdf5_sha256"]:raise RuntimeError("v467 HDF5 SHA drift")
  with h5py.File(source,"r") as h:cache[source]=np.asarray(h["joint_action/vector"],np.float32)
 actions=cache[source];start=int(row["start"]);history=actions[start:start+4];future=actions[start+4:start+12];anchor=history[-1,7:13];delta=future[:,7:13]-anchor;branches={"factual":future.copy()}
 for name,scale in (("no_transport",0.),("scale_0p4",.4),("scale_1p25",1.25),("reverse_direction_0p4",-.4)):
  value=future.copy();value[:,7:13]=anchor+scale*delta;branches[name]=value
 branches["factual_duplicate"]=future.copy();endpoint={n:float(np.linalg.norm(v[-1,7:13]-future[-1,7:13])) for n,v in branches.items() if n not in ("factual","factual_duplicate")};fullpath={n:float(np.linalg.norm(v[:,7:13]-future[:,7:13])) for n,v in branches.items() if n not in ("factual","factual_duplicate")}
 if arrsha(history)!=row["history_action_sha256"] or {n:arrsha(v) for n,v in branches.items()}!=row["branch_action_sha256"] or endpoint!=row["endpoint_diffs"] or fullpath!=row["path_diffs"] or min(endpoint.values())<.01 or min(fullpath.values())<.01:raise RuntimeError("v467 action closure drift")
 q0=history[-1,7:13];path=np.concatenate((q0[None],future[:,7:13]),0);step=np.linalg.norm(np.diff(path,axis=0),axis=1);total=float(step.sum());score=.5 if total<=1e-8 else float(step[:4].sum()/total);tie=hashlib.sha256(np.ascontiguousarray(np.concatenate((history,future),0),np.float32).view(np.uint8)).hexdigest();return score,tie
def shuffle_map(rows,cache):
 groups={}
 for row in rows:groups.setdefault(int(row["episode"]),[]).append(row)
 result={}
 for episode,group in sorted(groups.items()):
  evidence={id(row):request_phase(row,cache) for row in group};ordered=sorted(group,key=lambda row:evidence[id(row)]);n=len(ordered)
  if n<2:raise RuntimeError(f"v467 phase shuffle requires >=2 contexts episode={episode}")
  offset=max(1,n//2)
  for rank,row in enumerate(ordered):
   donor=ordered[(rank+offset)%n];result[f"{episode}:{int(row['start'])}"]={"donor_episode":episode,"donor_start":int(donor["start"]),"phase_score":evidence[id(row)][0],"donor_phase_score":evidence[id(donor)][0],"offset":offset,"group_size":n}
 return result
def schedule(ids,contexts,seed):
 rng=np.random.default_rng(seed);ordered=sorted(contexts);return [{"ids":rng.choice(ids,size=8,replace=False).tolist(),"context_id":ordered[step%len(ordered)]} for step in range(50)]
def schedule_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def main():
 p=argparse.ArgumentParser()
 for name in ("v456-final-receipt","v456-selection","v456-generation-report","v456-dataset","v169-release","v169-library","v169-runtime-source","trainer","runtime","packager","launcher","contract","output"):p.add_argument(f"--{name}",type=Path,required=True)
 a=p.parse_args();receipt=json.loads(a.v456_final_receipt.read_text());contract=json.loads(a.contract.read_text())
 if sha(a.contract)!="b9411f99492968fedc4e56cd046206ed037779dc1c451f2e08f7c2badb4ff394" or contract.get("format")!="strict-track2-v467-canonical-notransport-paired-delta-contract-v1" or contract.get("status")!="preregistered_public_train_s0_authorized":raise RuntimeError("v467 frozen contract/status drift")
 if receipt.get("format")!="strict-track2-v461-endpoint200-batch-audit-v1" or receipt.get("mode")!="final" or receipt.get("passed") is not True or not all(receipt.get("checks",{}).values()):raise RuntimeError("v467 blocked: v461 final")
 selection=json.loads(a.v456_selection.read_text());generation=json.loads(a.v456_generation_report.read_text())
 if selection.get("format")!="strict-track2-v461-endpoint200-preregistration-v1" or generation.get("passed") is not True or generation.get("exact_contexts")!=200:raise RuntimeError("v467 v461 contract")
 contexts=selection["contexts"];episodes=sorted({int(x["episode"]) for x in contexts});folds=[]
 for fold in range(5):
  held=sorted({int(x["episode"]) for x in contexts if int(x["fold"])==fold});fit=sorted(set(episodes)-set(held));folds.append({"fold":fold,"holdout_episodes":held,"fit_episodes":fit})
 if len(contexts)!=200 or len(episodes)!=15 or any(len(x["holdout_episodes"])!=3 for x in folds):raise RuntimeError("v467 folds")
 files=[]
 for batch in range(10):
  root=a.v456_dataset/f"batch_{batch:03d}";report=root/"batch_report.json";files.append({"relative":str(report.relative_to(a.v456_dataset)),"sha256":sha(report)})
  for row in sorted((root/"rows").iterdir()):
   for name in ("endpoint.npz","receipt.json"):
    path=row/name;files.append({"relative":str(path.relative_to(a.v456_dataset)),"sha256":sha(path)})
 if len(files)!=410:raise RuntimeError("v467 exact dataset tree")
 v169_manifest=a.v169_release/"v169_arm_routed_manifest.json";library_manifest=a.v169_library/"release_manifest.json"
 if not library_manifest.is_file():library_manifest=a.v169_library/"splits/adjust_bottle_50episodes_full.json"
 cache={};mappings=[]
 for fold in range(5):
  fit_rows=[x for x in contexts if int(x["fold"])!=fold];hold_rows=[x for x in contexts if int(x["fold"])==fold];mappings.append({"fold":fold,"fit_episodes":sorted({int(x["episode"]) for x in fit_rows}),"holdout_episodes":sorted({int(x["episode"]) for x in hold_rows}),"fit":shuffle_map(fit_rows,cache),"holdout":shuffle_map(hold_rows,cache)})
 ordered=[]
 for batch in range(10):ordered.extend(sorted((x for x in contexts if int(x["batch_id"])==batch),key=lambda x:f"episode{int(x['episode'])}_start{int(x['start']):05d}"))
 appearance_schedules=[];dynamics_schedules=[]
 for fold in range(5):
  fit_contexts=[i for i,x in enumerate(ordered) if int(x["fold"])!=fold];fit_ids=np.asarray([5*i+b for i in fit_contexts for b in range(5)]);appearance_schedules.append({"fold":fold,"sha256":schedule_sha(schedule(np.asarray(fit_contexts),set(fit_contexts),1816+fold))});dynamics_schedules.append({"fold":fold,"sha256":schedule_sha(schedule(fit_ids,set(fit_contexts),1616+fold)),"same_for_heads":["action","context_only","phase_shuffle"],"optimizer_independent":True})
 all_a_sha=schedule_sha(schedule(np.arange(200),set(range(200)),1916));all_d_sha=schedule_sha(schedule(np.arange(1000),set(range(200)),1716))
 payload={
  "format":"strict-track2-v467-canonical-notransport-paired-delta-preregistration-v1","created_at":datetime.now(timezone.utc).isoformat(),"seed":1616,"contract":{"path":str(a.contract.resolve()),"sha256":sha(a.contract)},
  "classification":"public-train parent-WM S0; no reward/policy/RL/dev/final/submission",
  "v461":{"final_receipt":{"path":str(a.v456_final_receipt.resolve()),"sha256":sha(a.v456_final_receipt)},"selection":{"path":str(a.v456_selection.resolve()),"sha256":sha(a.v456_selection)},"generation_report":{"path":str(a.v456_generation_report.resolve()),"sha256":sha(a.v456_generation_report)},"dataset":str(a.v456_dataset.resolve()),"files":files,"contexts":200,"samples":1000,"branches":["factual","no_transport","scale_0p4","scale_1p25","reverse_direction_0p4"]},
  "folds":folds,"phase_shuffle":{"formula":"request-only phase score; within-episode half rotation; no RGB/reward/outcome","mappings":mappings},
  "v169":{"release":str(a.v169_release.resolve()),"library":str(a.v169_library.resolve()),"runtime_source":{"path":str(a.v169_runtime_source.resolve()),"sha256":sha(a.v169_runtime_source)},"release_manifest_sha256":sha(v169_manifest),"library_manifest":str(library_manifest.resolve()),"library_manifest_sha256":sha(library_manifest),"release_files":release_closure(a.v169_release,a.v169_release.resolve().parent),"library_files":library_closure(a.v169_release,a.v169_library)},
  "source":{"trainer_path":str(a.trainer.resolve()),"trainer_sha256":sha(a.trainer),"runtime_path":str(a.runtime.resolve()),"runtime_sha256":sha(a.runtime),"packager_path":str(a.packager.resolve()),"packager_sha256":sha(a.packager),"launcher_path":str(a.launcher.resolve()),"launcher_sha256":sha(a.launcher),"contract_path":str(a.contract.resolve()),"contract_sha256":sha(a.contract)},
  "decomposition":{"canonical_future":"requested float32 copy with [:,7:13]=history[-1,7:13]","appearance_target":"Y_no_transport-B0","dynamics_target":"Y_branch-Y_no_transport","no_transport_dynamics_exact_zero":True,"comparator_input_to_controls":False},
  "training":{"appearance_heads_per_fold":1,"dynamics_heads_per_fold":["action","context_only","phase_shuffle"],"steps":50,"batch_size":8,"channels":16,"precision":"bf16","learning_rate":0.0003,"weight_decay":0.0001,"gradient_clip":1.0,"appearance_loss":[1.0,0.25],"dynamics_loss":[1.0,0.25,0.5]},
  "kill_gate":contract["unchanged_kill_gates"],"early_stop":contract["early_stop"],"common_v169_seed":"sha256('v466/1616/{episode}/{start}')[:8] little-endian mod 2**31; branch-independent",
  "appearance_schedules":appearance_schedules,"dynamics_schedules":dynamics_schedules,"all200_appearance_schedule_sha256":all_a_sha,"all200_dynamics_schedule_sha256":all_d_sha,
  "runtime":contract["runtime"],"guards":{"reward_loaded":False,"policy_updates":0,"rl_authorized":False,"development_or_final_used":False,"real_submission":False}
 }
 if a.output.exists() or a.output.with_name(a.output.name+".tmp").exists():raise RuntimeError("refusing overwrite v467 prereg")
 a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_name(a.output.name+".tmp")
 with tmp.open("x") as handle:json.dump(payload,handle,indent=2);handle.write("\n");handle.flush();os.fsync(handle.fileno())
 os.replace(tmp,a.output);fd=os.open(a.output.parent,os.O_RDONLY);os.fsync(fd);os.close(fd);print(a.output);return 0
if __name__=="__main__":raise SystemExit(main())
