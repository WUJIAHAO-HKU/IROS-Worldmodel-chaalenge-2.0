#!/usr/bin/env python3
"""Preregister four public-train postclose endpoint interventions for v455."""
from __future__ import annotations
import argparse, ast, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import h5py, numpy as np

SEED=1608
VARIANTS=("factual","no_transport","scale_0p4","scale_1p25","reverse_direction_0p4","factual_duplicate")
EXPECTED_GOLDEN=((32,44,95),(12,14,100),(44,57,103),(42,55,105))
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(8<<20),b""):h.update(block)
 return h.hexdigest()
def function_sha(path,name):
 text=Path(path).read_text(); tree=ast.parse(text); node=next(x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name==name)
 return hashlib.sha256(ast.get_source_segment(text,node).encode()).hexdigest()
def make_branches(history,future):
 anchor=history[-1,7:13]; delta=future[:,7:13]-anchor; result={}
 for name,scale in (("no_transport",0.),("scale_0p4",.4),("scale_1p25",1.25),("reverse_direction_0p4",-.4)):
  value=future.copy(); value[:,7:13]=anchor+scale*delta; result[name]=value
 return {"factual":future.copy(),**result,"factual_duplicate":future.copy()}
def main():
 p=argparse.ArgumentParser()
 for name in ("split","dataset","windows","seed-file","collector","auditor","resize-source","output"):p.add_argument(f"--{name}",type=Path,required=True)
 a=p.parse_args(); split=json.loads(a.split.read_text()); train=set(map(int,split["train_episodes"])); validation=set(map(int,split["validation_episodes"])); arms={int(k):v for k,v in split["arm_by_episode"].items()}; prompts={int(k):v for k,v in split["episode_to_instruction"].items()}; seeds=list(map(int,a.seed_file.read_text().split()))
 if len(seeds)!=50:raise RuntimeError("v455 public seed contract failed")
 train_actions=[]; per_episode={}
 for episode in sorted(train):
  source=a.dataset/f"data/episode{episode}.hdf5"
  with h5py.File(source,"r") as h5:value=np.asarray(h5["joint_action/vector"],dtype=np.float32)
  train_actions.append(value);per_episode[episode]=(source,value)
 train_actions=np.concatenate(train_actions); lower=train_actions.min(0); upper=train_actions.max(0); contexts=[]
 candidates=[]
 for episode in sorted(e for e in train if arms[e]=="right"):
  source,value=per_episode[episode];options=[]
  for start in range(0,len(value)-11):
   history=value[start:start+4];future=value[start+4:start+12]
   if np.all(future[:,13]<.5):options.append((float(np.max(np.linalg.norm(future[:,7:13]-history[-1,7:13],axis=1))),start))
  if not options:raise RuntimeError(f"v455 no postclose candidate episode{episode}")
  metric,start=sorted(options,key=lambda x:(-x[0],x[1]))[0]
  closed=np.flatnonzero(value[:,13]<.5)
  if len(closed)==0 or len(value)-1<=int(closed[0]):raise RuntimeError(f"v455 invalid first-close phase episode{episode}")
  first_close=int(closed[0]);future_start=start+4;phase=float((future_start-first_close)/(len(value)-1-first_close))
  candidates.append({"episode":episode,"start":start,"right6_max_path_l2":metric,"first_right_close":first_close,"future_start":future_start,"phase":phase,"source_hdf5":str(source.resolve()),"source_hdf5_sha256":sha(source)})
 candidates.sort(key=lambda x:(x["right6_max_path_l2"],x["episode"]));n_candidates=len(candidates)
 if n_candidates!=15:raise RuntimeError(f"v455 expected 15 public-right candidates, got {n_candidates}")
 for rank0,row in enumerate(candidates):row["rank0_asc"]=rank0;row["quartile"]=1+min(3,(4*rank0)//n_candidates)
 phase_targets=np.linspace(min(x["phase"] for x in candidates),max(x["phase"] for x in candidates),4)
 selected=[]
 for quartile,target in enumerate(phase_targets,1):
  pool=[x for x in candidates if x["quartile"]==quartile]
  selected.append(min(pool,key=lambda x:(abs(x["phase"]-float(target)),x["episode"])))
 fixed=tuple((x["episode"],seeds[x["episode"]],x["start"]) for x in selected);chosen=[x[0] for x in fixed]
 if fixed!=EXPECTED_GOLDEN:raise RuntimeError(f"v455 deterministic public-action selection drift: {fixed} != {EXPECTED_GOLDEN}")
 if {x["quartile"] for x in selected}!={1,2,3,4} or len({x["episode"] for x in selected})!=4:raise RuntimeError("v455 quartile selection contract failed")
 if not set(chosen)<=train or set(chosen)&validation or any(arms[e]!="right" for e in chosen):raise RuntimeError("v455 public-right train contract failed")
 for episode,sim_seed,start in fixed:
  source=a.dataset/f"data/episode{episode}.hdf5"; window=a.windows/f"episode{episode}_{start:05d}.npz"
  with h5py.File(source,"r") as h5:actions=np.asarray(h5["joint_action/vector"],dtype=np.float32)
  with np.load(window,allow_pickle=False) as z:public_history=z["history_actions"].astype(np.float32); public_future=z["future_actions"].astype(np.float32)
  if start<0 or start+12>len(actions) or seeds[episode]!=sim_seed or not np.array_equal(public_history,actions[start:start+4]) or not np.array_equal(public_future,actions[start+4:start+12]):raise RuntimeError(f"v455 fixed source drift episode{episode}")
  history=actions[start:start+4]; future=actions[start+4:start+12]; branches=make_branches(history,future); anchor=history[-1,7:13]
  if not np.all(future[:,13]<.5):raise RuntimeError(f"v455 gripper not postclose episode{episode}")
  endpoint_diff={name:float(np.linalg.norm(value[-1,7:13]-future[-1,7:13])) for name,value in branches.items() if name not in ("factual","factual_duplicate")}
  bounds={name:bool(np.all(value[:,7:13]>=lower[7:13]) and np.all(value[:,7:13]<=upper[7:13])) for name,value in branches.items()}
  if min(endpoint_diff.values())<.01 or not all(bounds.values()) or not all(np.array_equal(value[:,:7],future[:,:7]) and np.array_equal(value[:,13],future[:,13]) for value in branches.values()):raise RuntimeError(f"v455 branch action contract failed episode{episode}: {endpoint_diff} {bounds}")
  contexts.append({"episode":episode,"dataset_seed":sim_seed,"start":start,"instruction":prompts[episode],"length":len(actions),"right6_max_path_l2":float(np.max(np.linalg.norm(future[:,7:13]-anchor,axis=1))),"endpoint_diff_l2":endpoint_diff,"source_hdf5":str(source.resolve()),"source_hdf5_sha256":sha(source),"source_window":str(window.resolve()),"source_window_sha256":sha(window)})
 selected_map={(row["episode"],row["start"]):row for row in candidates};fixed_pairs={(episode,start) for episode,_seed,start in fixed}
 if not fixed_pairs<=set(selected_map):raise RuntimeError(f"v455 selected contexts are not per-episode max-path candidates: {fixed_pairs-set(selected_map)}")
 preprocessing={"function":"wam_pipeline.data.resize_rgb","size":256,"mode":"RGB","interpolation":"PIL.Image.Resampling.BILINEAR","source":str(a.resize_source.resolve()),"source_sha256":sha(a.resize_source),"function_source_sha256":function_sha(a.resize_source,"resize_rgb")}
 files={"split":a.split,"seed_file":a.seed_file,"collector":a.collector,"auditor":a.auditor,"resize_source":a.resize_source}
 payload={"format":"strict-track2-v455-postclose-endpoint-pilot-preregistration-v1","registered_at":datetime.now(timezone.utc).isoformat(),"seed":SEED,"classification":"public-train within-simulator paired endpoint residual parent data only","public_train":{"episodes":chosen,"validation_excluded":sorted(validation),"contexts":contexts,"per_dim_action_lower":lower.tolist(),"per_dim_action_upper":upper.tolist()},"selection_manifest":{"phase_target":"four targets linearly spaced over observed phase range","phase_formula":"(future_start-first_right_close)/(episode_length-1-first_right_close)","phase_targets":phase_targets.tolist(),"population":"all public train40 episodes routed right (15)","population_count":n_candidates,"rank_order":"right6_max_path_l2 ascending, episode ascending tie break","quartile_formula":"1+min(3,floor(4*rank0/15))","selection_rule":"within each quartile minimize abs(phase-phase_target), episode ascending tie break","candidates":candidates,"fixed_selected":selected,"expected_golden":[list(x) for x in EXPECTED_GOLDEN],"split_path":str(a.split.resolve()),"split_sha256":sha(a.split)},"indexing":{"prefix":"actions[:start]","history":"actions[start:start+4]","future":"actions[start+4:start+12]"},"preprocessing":preprocessing,"branches":{"ordered":list(VARIANTS),"factual":"future8","no_transport":"anchor+0.0*(future-anchor)","scale_0p4":"anchor+0.4*(future-anchor)","scale_1p25":"anchor+1.25*(future-anchor)","reverse_direction_0p4":"anchor-0.4*(future-anchor)","factual_duplicate":"independent factual replay","modified_columns":[7,8,9,10,11,12],"gripper_unchanged":True},"execution":{"planner_backend":"mplib","context_workers":2,"branches_serial_within_context":True,"future_single_chunk_call":True,"endpoint_only":True,"intermediate_frames_fabricated":False,"success_check_disabled":True,"reward_success_done_read":False,"retry":0,"context_timeout_seconds":300,"wall_time_max_minutes":20,"gpu_peak_mib_max":24576,"output_bytes_max":16777216},"technical_gate":{"same_context_bitexact":True,"factual_duplicate_rgb_mae_max":0.5,"action_endpoint_diff_l2_min":0.01,"counterfactual_endpoint_rgb_mae_min":1.0,"counterfactual_endpoint_qpos_l2_min":0.01,"counterfactual_bottle_position_l2_min":0.005,"counterfactuals_per_context_min":3,"contexts_per_transport_min":3,"factual_public_endpoint_mae":"diagnostic_only"},"evidence_sha256":{k:sha(v) for k,v in files.items()},"guards":{"paired_endpoint_residual_only":True,"factual_public_fidelity_diagnostic_only":True,"development_or_final_used":False,"task_outcome_or_reward_selection":False,"all_fixed_rows_branches_retained":True,"policy_bc_target":False,"policy_updates":0,"reward_loaded":False,"endpoint_parent_data_authorized_if_pass":True,"rl_authorized":False,"real_submission":False}}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(payload,indent=2)+"\n"); print(a.output); return 0
if __name__=="__main__":raise SystemExit(main())
