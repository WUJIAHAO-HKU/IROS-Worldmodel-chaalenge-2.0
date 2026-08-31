#!/usr/bin/env python3
"""Preregister four public-right, train-only RoboTwin paired interventions."""
from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import h5py, numpy as np
from wam_pipeline.v442_v169_close_aligned_projection_runtime import gate_decision

SEED=1604; VARIANTS=("factual","open","static_transport","reverse_transport","hold_all","factual_duplicate")
FIXED=((28,38,56,0,"base16"),(15,18,54,2,"base16"),(25,34,49,4,"base13"),(30,40,55,6,"base13"))
def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""): h.update(b)
 return h.hexdigest()
def main()->int:
 p=argparse.ArgumentParser()
 for n in ("split","dataset","windows","seed-file","collector","auditor","gate-runtime","output"): p.add_argument(f"--{n}",type=Path,required=True)
 a=p.parse_args(); split=json.loads(a.split.read_text()); train=set(map(int,split["train_episodes"])); validation=set(map(int,split["validation_episodes"])); arms={int(k):v for k,v in split["arm_by_episode"].items()}; prompts={int(k):v for k,v in split["episode_to_instruction"].items()}
 right=sorted(e for e in train if arms[e]=="right"); chosen=[x[0] for x in FIXED]; seeds=list(map(int,a.seed_file.read_text().split())); scene=json.loads((a.dataset/"scene_info.json").read_text())
 if len(seeds)!=50 or not set(chosen)<=set(right) or set(chosen)&validation: raise RuntimeError("v451 public-right train seed contract failed")
 contexts=[]
 for e,sim_seed,start,phase,base in FIXED:
  window=a.windows/f"episode{e}_{start:05d}.npz"; source=a.dataset/f"data/episode{e}.hdf5"
  with np.load(window,allow_pickle=False) as z: history=z["history_actions"].astype(np.float32); future=z["future_actions"].astype(np.float32); d=gate_decision(history,future,prompts[e])
  with h5py.File(source,"r") as h5: actions=np.asarray(h5["joint_action/vector"],dtype=np.float32); length=len(actions)
  actual_base=str(scene[f"episode_{e}"]["info"]["{A}"]).split("/")[-1]
  if start<0 or start+12>length or not np.array_equal(history,actions[start:start+4]) or not np.array_equal(future,actions[start+4:start+12]) or seeds[e]!=sim_seed or not bool(d.get("gate")) or int(d.get("first_close_index"))!=phase or actual_base!=base: raise RuntimeError(f"v451 fixed context drift episode{e}: seed={seeds[e]} phase={d} base={actual_base}")
  contexts.append({"episode":e,"dataset_seed":sim_seed,"start":start,"phase":phase,"bottle_base":base,"length":length,"instruction":prompts[e],"source_hdf5":str(source.resolve()),"source_hdf5_sha256":sha(source),"source_window":str(window.resolve()),"source_window_sha256":sha(window)})
 files={"split":a.split,"seed_file":a.seed_file,"collector":a.collector,"auditor":a.auditor,"gate_runtime":a.gate_runtime}
 payload={"format":"strict-track2-v451-paired-intervention-pilot-preregistration-v1","registered_at":datetime.now(timezone.utc).isoformat(),"seed":SEED,"classification":"technical train-only simulator pilot; no model/policy authority","public_train":{"episodes":chosen,"validation_excluded":sorted(validation),"contexts":contexts},"indexing":{"prefix":"actions[:start]","history":"actions[start:start+4]","future":"actions[start+4:start+12]","source_window_start_matches_start":True},"branches":{"ordered":list(VARIANTS),"factual":"source future8","open":"factual with right gripper column13=1 for all8","static_transport":"factual with right qpos columns7:13 held at history[-1]","reverse_transport":"factual with right qpos columns7:13 reversed; factual grippers retained","hold_all":"repeat full history[-1] for8","factual_duplicate":"independent reset/replay of factual"},"execution":{"planner_backend":"mplib","curobo_required":False,"context_workers":2,"branches_serial_within_context":True,"same_dataset_seed_reset_each_branch":True,"same_prefix_and_history_replay":True,"context_frames":5,"future_steps":8,"success_check_disabled_only_to_enforce_fixed_horizon":True,"reward_success_done_read":False,"open3d_import_required_mode":"native","open3d_stub_rgb_only":False,"pointcloud_calls":0,"runtime_provenance_required":["sys.executable","sys.prefix","open3d.file_version","toppra.file_version","mplib.file_version","sapien.file_version"],"optional_curobo_traceback_nonfatal":True,"technical_intervention_effect_gate":True,"task_outcome_or_reward_selection":False,"all_fixed_rows_branches_retained":True,"exact_contexts":4,"exact_branches_per_context":6,"retry":0,"context_timeout_seconds":300,"wall_time_max_minutes":20,"output_bytes_max":16777216,"gpu_peak_mib_max":16384},"technical_gate":{"context_rgb_bitexact_or_mae_max":0.25,"context_qpos_max_abs":1e-5,"context_pose_max_abs":1e-5,"context_bottle_position_max_abs":1e-5,"factual_duplicate_rgb_mae_max":0.5,"each_factual_replay_public_target_rgb_mae_max":8.0,"causal_contexts_min":3,"causal_counterfactuals_per_context_min":2,"counterfactual_final_rgb_mae_min_floor":1.0,"counterfactual_final_rgb_mae_noise_multiple":5.0,"counterfactual_final_qpos_l2_min":0.01,"counterfactual_bottle_position_l2_min":0.005},"evidence_sha256":{k:sha(v) for k,v in files.items()},"guards":{"open3d_import_required_mode":"native","open3d_stub_rgb_only":False,"pointcloud_calls":0,"runtime_provenance_required":["sys.executable","sys.prefix","open3d.file_version","toppra.file_version","mplib.file_version","sapien.file_version"],"optional_curobo_traceback_nonfatal":True,"technical_intervention_effect_gate":True,"task_outcome_or_reward_selection":False,"all_fixed_rows_branches_retained":True,"development_or_final_used":False,"hidden_or_official_payload_used":False,"policy_bc_target":False,"policy_updates":0,"reward_loaded":False,"real_submission":False,"s1_authorized":False,"rl_authorized":False}}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(payload,indent=2)+"\n"); print(a.output); return 0
if __name__=="__main__": raise SystemExit(main())
