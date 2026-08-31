#!/usr/bin/env python3
"""Audit v455 paired endpoint data and authorization boundary."""
from __future__ import annotations
import argparse, ast, hashlib, importlib, json, sys
from importlib import metadata
from pathlib import Path
import h5py, numpy as np

VARIANTS=("factual","no_transport","scale_0p4","scale_1p25","reverse_direction_0p4","factual_duplicate");TRANSPORT=VARIANTS[1:5]
def digest(x):return hashlib.sha256(np.ascontiguousarray(x).view(np.uint8)).hexdigest()
def file_digest(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(8<<20),b""):h.update(block)
 return h.hexdigest()
def function_digest(path,name):
 text=Path(path).read_text();tree=ast.parse(text);node=next(x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name==name);return hashlib.sha256(ast.get_source_segment(text,node).encode()).hexdigest()
def provenance():
 modules={}
 for name in ("open3d","toppra","mplib","sapien"):
  module=importlib.import_module(name);version=getattr(module,"__version__",None)
  if version is None:
   try:version=metadata.version(name)
   except metadata.PackageNotFoundError:version="unknown"
  modules[name]={"file":str(Path(module.__file__).resolve()),"version":str(version)}
 return {"sys_executable":sys.executable,"sys_prefix":sys.prefix,"modules":modules}
def expected_branches(history,future):
 anchor=history[-1,7:13];delta=future[:,7:13]-anchor;out={}
 for name,scale in (("no_transport",0.),("scale_0p4",.4),("scale_1p25",1.25),("reverse_direction_0p4",-.4)):
  value=future.copy();value[:,7:13]=anchor+scale*delta;out[name]=value
 return np.stack([future,out["no_transport"],out["scale_0p4"],out["scale_1p25"],out["reverse_direction_0p4"],future])
def main():
 p=argparse.ArgumentParser()
 for name in ("preregistration","dataset-dir","generator","resize-source","output"):p.add_argument(f"--{name}",type=Path,required=True)
 a=p.parse_args();pre=json.loads(a.preregistration.read_text());rep=json.loads((a.dataset_dir/"generation_report.json").read_text());source=a.generator.read_text();tree=ast.parse(source);specs={(int(x["episode"]),int(x["start"])):x for x in pre["public_train"]["contexts"]};files=sorted(a.dataset_dir.glob("episode*_start*.npz"));observed=set();rows=[];counts={name:0 for name in TRANSPORT};lower=np.asarray(pre["public_train"]["per_dim_action_lower"],np.float32);upper=np.asarray(pre["public_train"]["per_dim_action_upper"],np.float32)
 for path in files:
  with np.load(path,allow_pickle=False) as z:
   episode=int(z["episode"]);start=int(z["start"]);observed.add((episode,start));spec=specs[(episode,start)];variants=list(map(str,z["variants"]));contexts=z["branch_context_rgb"];states=z["branch_context_state"];poses=z["branch_context_pose"];bottles=z["branch_context_bottle_position"];history=z["history_actions"];futures=z["future_actions"];endpoint=z["endpoint_rgb"];end_state=z["endpoint_state"];end_bottle=z["endpoint_bottle_position"];keys=set(z.files)
   expected=expected_branches(history,futures[0]);action_diff={name:float(np.linalg.norm(futures[i,-1,7:13]-futures[0,-1,7:13])) for i,name in enumerate(TRANSPORT,1)};context_exact=all(np.array_equal(contexts[0],contexts[i]) and np.array_equal(states[0],states[i]) and np.array_equal(poses[0],poses[i]) and np.array_equal(bottles[0],bottles[i]) for i in range(1,6));duplicate=float(np.abs(endpoint[0].astype(np.float32)-endpoint[5].astype(np.float32)).mean());effects={}
   for i,name in enumerate(TRANSPORT,1):
    rgb=float(np.abs(endpoint[i].astype(np.float32)-endpoint[0].astype(np.float32)).mean());qpos=float(np.linalg.norm(end_state[i]-end_state[0]));obj=float(np.linalg.norm(end_bottle[i]-end_bottle[0]));passed=rgb>=1. and (qpos>=.01 or obj>=.005);effects[name]={"rgb":rgb,"qpos":qpos,"bottle":obj,"passed":passed};counts[name]+=int(passed)
   with np.load(spec["source_window"],allow_pickle=False) as w:public=w["target_frames"][-1].astype(np.uint8);public_history=w["history_actions"].astype(np.float32);public_future=w["future_actions"].astype(np.float32)
   public_mae=float(np.abs(endpoint[0].astype(np.float32)-public.astype(np.float32)).mean())
   with h5py.File(spec["source_hdf5"],"r") as h5:actions=np.asarray(h5["joint_action/vector"],np.float32)
   checks={"schema_shapes":variants==list(VARIANTS) and contexts.shape==(6,256,256,3) and contexts.dtype==np.uint8 and endpoint.shape==(6,256,256,3) and endpoint.dtype==np.uint8 and futures.shape==(6,8,14) and history.shape==(4,14) and "target_frames" not in keys,"source_closure":file_digest(spec["source_hdf5"])==spec["source_hdf5_sha256"] and file_digest(spec["source_window"])==spec["source_window_sha256"],"source_binding":np.array_equal(history,actions[start:start+4]) and np.array_equal(futures[0],actions[start+4:start+12]) and np.array_equal(history,public_history) and np.array_equal(futures[0],public_future),"branch_formula":np.allclose(futures,expected,atol=1e-7,rtol=0) and np.array_equal(futures[:,:,:7],np.repeat(futures[0:1,:,:7],6,0)) and np.array_equal(futures[:,:,13],np.repeat(futures[0:1,:,13],6,0)),"action_bounds_diff":bool(np.all(futures>=lower) and np.all(futures<=upper)) and min(action_diff.values())>=.01,"context_bitexact_hashes":context_exact and list(map(str,z["context_rgb_sha256"]))==[digest(x) for x in contexts] and list(map(str,z["context_state_sha256"]))==[digest(x) for x in states],"executed_binding":list(map(str,z["executed_action_sha256"]))==[digest(x) for x in futures],"duplicate":duplicate<=.5,"effect_three_of_four":sum(x["passed"] for x in effects.values())>=3,"public_endpoint_diagnostic_only":digest(public)==str(z["public_factual_endpoint_sha256"]) and abs(public_mae-float(z["factual_public_endpoint_rgb_mae"]))<1e-9,"no_intermediate_or_outcome":not keys.intersection({"reward","success","done","outcome","target_frames","intermediate_frames"})}
   rows.append({"episode":episode,"start":start,"passed":all(checks.values()),"checks":checks,"action_diff":action_diff,"effects":effects,"factual_public_endpoint_rgb_mae_diagnostic":public_mae})
 guards=rep.get("guards",{});preproc=pre["preprocessing"];actual_prov=provenance();imports=[n.module or "" for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]+[x.name for n in ast.walk(tree) if isinstance(n,ast.Import) for x in n.names]
 checks={"format":pre.get("format")=="strict-track2-v455-postclose-endpoint-pilot-preregistration-v1" and rep.get("format")=="strict-track2-v455-postclose-endpoint-pilot-generation-report-v1","generation_pass":rep.get("passed") is True and not rep.get("technical_failures"),"exact_fixed_four":len(files)==4 and observed==set(specs) and all(x["passed"] for x in rows),"transport_each_three_contexts":all(counts[name]>=3 for name in TRANSPORT) and counts==rep.get("transport_effect_context_counts"),"single_chunk_endpoint_source":source.count("env.step(action_map[name][None])")==1 and "endpoint_rgb=end_rgb" in source and "prediction.append" not in source,"preprocess_closure":preproc["source_sha256"]==file_digest(a.resize_source) and preproc["function_source_sha256"]==function_digest(a.resize_source,"resize_rgb") and pre.get("evidence_sha256",{}).get("resize_source")==file_digest(a.resize_source) and "from wam_pipeline.data import resize_rgb" in source,"runtime_provenance":rep.get("runtime_provenance")==actual_prov,"resource_contract":rep.get("gpu_peak_mib",2**60)<=24576 and rep.get("wall_seconds",2**60)<=1200 and rep.get("output_bytes",2**60)<=16777216,"endpoint_authority_only":guards.get("future_single_chunk_call") is True and guards.get("endpoint_only") is True and guards.get("intermediate_frames_fabricated") is False and guards.get("factual_public_endpoint_mae_diagnostic_only") is True and guards.get("endpoint_parent_data_authorized_if_pass") is True and guards.get("rl_authorized") is False,"no_reward_policy":not any("reward" in x.lower() for x in imports) and not any(token in source.lower() for token in ("optimizer","backward(","policy_action","mpc"))}
 manifest=pre.get("selection_manifest",{});split=json.loads(Path(manifest.get("split_path","")).read_text());arms={int(k):v for k,v in split["arm_by_episode"].items()};train=set(map(int,split["train_episodes"]));right=sorted(e for e in train if arms.get(e)=="right");recorded_by_episode={int(x["episode"]):x for x in manifest.get("candidates",[])};recomputed=[]
 for episode in right:
  recorded=recorded_by_episode.get(episode,{});source_path=Path(recorded.get("source_hdf5","/__missing_v455_source__"))
  if not source_path.is_file() or file_digest(source_path)!=recorded.get("source_hdf5_sha256"):continue
  with h5py.File(source_path,"r") as h5:value=np.asarray(h5["joint_action/vector"],np.float32)
  options=[]
  for start in range(0,len(value)-11):
   history=value[start:start+4];future=value[start+4:start+12]
   if np.all(future[:,13]<.5):options.append((float(np.max(np.linalg.norm(future[:,7:13]-history[-1,7:13],axis=1))),start))
  metric,start=sorted(options,key=lambda x:(-x[0],x[1]))[0];closed=np.flatnonzero(value[:,13]<.5)
  if not len(closed) or len(value)-1<=int(closed[0]):continue
  first_close=int(closed[0]);future_start=start+4;phase=float((future_start-first_close)/(len(value)-1-first_close))
  recomputed.append({"episode":episode,"start":start,"right6_max_path_l2":metric,"first_right_close":first_close,"future_start":future_start,"phase":phase,"source_hdf5":str(source_path.resolve()),"source_hdf5_sha256":file_digest(source_path)})
 recomputed.sort(key=lambda x:(x["right6_max_path_l2"],x["episode"]));n=len(recomputed)
 for rank0,row in enumerate(recomputed):row["rank0_asc"]=rank0;row["quartile"]=1+min(3,(4*rank0)//15)
 phase_targets=np.linspace(min((x["phase"] for x in recomputed),default=0.),max((x["phase"] for x in recomputed),default=0.),4)
 independently_selected=[]
 for quartile,target in enumerate(phase_targets,1):
  pool=[x for x in recomputed if x["quartile"]==quartile]
  if pool:independently_selected.append(min(pool,key=lambda x:(abs(x["phase"]-float(target)),x["episode"])))
 fixed_manifest={(int(x["episode"]),int(x["start"])) for x in manifest.get("fixed_selected",[])};fixed_context={(int(x["episode"]),int(x["start"])) for x in pre["public_train"]["contexts"]}
 expected_golden=((32,44,95),(12,14,100),(44,57,103),(42,55,105));selected_pairs={(x["episode"],x["start"]) for x in independently_selected};golden_pairs={(e,s) for e,_seed,s in expected_golden};context_triplets={(int(x["episode"]),int(x["dataset_seed"]),int(x["start"])) for x in pre["public_train"]["contexts"]}
 checks["selection_manifest_exact"]=len(right)==n==manifest.get("population_count")==15 and set(recorded_by_episode)==set(right) and file_digest(manifest["split_path"])==manifest["split_sha256"] and recomputed==manifest.get("candidates") and np.allclose(np.asarray(manifest.get("phase_targets",[])),phase_targets,atol=0,rtol=0) and manifest.get("rank_order")=="right6_max_path_l2 ascending, episode ascending tie break" and manifest.get("quartile_formula")=="1+min(3,floor(4*rank0/15))" and {x["quartile"] for x in independently_selected}=={1,2,3,4} and selected_pairs==fixed_manifest==fixed_context==set(specs)==golden_pairs and context_triplets==set(expected_golden) and manifest.get("expected_golden")==[list(x) for x in expected_golden]
 checks["paired_endpoint_residual_only"]="paired endpoint residual parent data only" in pre.get("classification","") and pre.get("guards",{}).get("paired_endpoint_residual_only") is True and guards.get("paired_endpoint_residual_only") is True and guards.get("factual_public_fidelity_diagnostic_only") is True and guards.get("rl_authorized") is False
 passed=all(checks.values());audit={"format":"strict-track2-v455-postclose-endpoint-pilot-audit-v1","passed":passed,"checks":checks,"rows":rows,"transport_effect_context_counts":counts,"selection_manifest":manifest,"runtime_provenance":actual_prov,"guards":{"paired_endpoint_residual_only":True,"factual_public_fidelity_diagnostic_only":True,"endpoint_parent_data_authorized":passed,"rl_authorized":False,"policy_updates":0,"real_submission":False}}
 a.output.write_text(json.dumps(audit,indent=2)+"\n");print(json.dumps(audit,indent=2));return 0 if passed else 2
if __name__=="__main__":raise SystemExit(main())
