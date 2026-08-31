#!/usr/bin/env python3
"""Independent immutable auditor for the v477 temporal paired-simulator pilot."""
from __future__ import annotations
import argparse,ast,hashlib,json,os
from pathlib import Path
import h5py,numpy as np

PRE_FORMAT="strict-track2-v477-public-train-paired-temporal8-pilot-preregistration-v1"
REPORT_FORMAT="strict-track2-v477-public-train-paired-temporal8-pilot-generation-report-v1"
AUDIT_FORMAT="strict-track2-v477-public-train-paired-temporal8-pilot-audit-v1"
CONTRACT_SHA="9fbb494f55115786b2b22e77d91c172ec1a52df77cec224083ff6e82391e53a1"
VARIANTS=("factual","no_transport","scale_0p4","scale_1p25","reverse_direction_0p4","factual_duplicate");TRANSPORT=VARIANTS[1:5]
KEYS={"episode","dataset_seed","start","variants","prefix_lengths","instruction","pre_future_context_rgb","pre_future_context_state","pre_future_context_pose","pre_future_context_bottle_position","history_actions","future_actions","temporal_rgb","temporal_state","temporal_pose","temporal_bottle_position","context_rgb_sha256","context_state_sha256","executed_prefix_action_sha256","public_factual_target_rgb","public_factual_target_sha256","factual_public_frame_rgb_mae"}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def arrsha(x):return hashlib.sha256(np.ascontiguousarray(x).view(np.uint8)).hexdigest()
def support_tree(root):
 root=Path(root).resolve();paths=sorted(p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts and p.suffix.lower() in (".py",".yml",".yaml"));items=[[p.relative_to(root).as_posix(),sha(p)] for p in paths]
 return hashlib.sha256(json.dumps(items,sort_keys=True,separators=(",",":")).encode()).hexdigest(),len(items)
def branches(history,future):
 anchor=history[-1,7:13];delta=future[:,7:13]-anchor;out={}
 for name,scale in (("no_transport",0.),("scale_0p4",.4),("scale_1p25",1.25),("reverse_direction_0p4",-.4)):
  value=future.copy();value[:,7:13]=anchor+scale*delta;out[name]=value
 return {"factual":future.copy(),**out,"factual_duplicate":future.copy()}
def atomic(path,obj):
 path=Path(path);tmp=path.with_name(path.name+".tmp")
 if path.exists() or tmp.exists():raise FileExistsError(path)
 with tmp.open("x",encoding="utf-8") as f:json.dump(obj,f,sort_keys=True,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
 os.replace(tmp,path);fd=os.open(str(path.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
def equal_context(x,oracle):return all(np.array_equal(x[i],np.repeat(oracle[i,None],8,axis=0)) for i in range(6))
def main():
 ap=argparse.ArgumentParser()
 for name in ("preregistration","collector","dataset","generation-report","support-root","task-config","resize-source","output"):ap.add_argument("--"+name,required=True,type=Path)
 a=ap.parse_args();pre=json.loads(a.preregistration.read_text());report=json.loads(a.generation_report.read_text())
 closure=pre.get("execution_closure",{});support_digest,support_count=support_tree(a.support_root)
 closure_digest=hashlib.sha256(json.dumps(closure,sort_keys=True,separators=(",",":")).encode()).hexdigest()
 pre_ok=pre.get("format")==PRE_FORMAT and pre.get("status")=="preregistered_public_train_paired_temporal8_pilot_authorized" and pre.get("contract",{}).get("sha256")==CONTRACT_SHA and sha(Path(__file__).resolve())==pre.get("evidence_sha256",{}).get("auditor") and report.get("format")==REPORT_FORMAT and report.get("passed") is True and report.get("preregistration_sha256")==sha(a.preregistration) and report.get("collector_sha256")==sha(a.collector)==pre.get("evidence_sha256",{}).get("collector") and report.get("generator_sha256")==pre.get("evidence_sha256",{}).get("generator") and report.get("execution_closure_digest")==closure_digest and Path(closure["support_root"]["path"]).resolve()==a.support_root.resolve() and support_digest==closure["support_root"]["source_config_tree_sha256"]==report.get("support_source_config_tree_sha256") and support_count==closure["support_root"]["source_config_file_count"] and Path(closure["task_config"]["path"]).resolve()==a.task_config.resolve() and sha(a.task_config)==closure["task_config"]["sha256"]==report.get("task_config_sha256") and Path(closure["resize_source"]["path"]).resolve()==a.resize_source.resolve() and sha(a.resize_source)==closure["resize_source"]["sha256"]==report.get("resize_source_sha256") and report.get("runtime_provenance")==pre.get("runtime_provenance_expected") and pre.get("guards",{}).get("technical_effect_is_not_task_outcome_selection") is True and pre.get("guards",{}).get("all_fixed_rows_branches_retained") is True
 source=Path(a.collector).read_text();tree=ast.parse(source);forbidden_calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="check_success"]
 source_guard=not forbidden_calls and "gen_sparse_reward_data" not in source and "save_pcd" not in source
 specs={(int(x["episode"]),int(x["start"])):x for x in pre["contexts"]};expected_order=list(specs);rows=[];named={n:0 for n in TRANSPORT};row_checks=[]
 for key,spec in specs.items():
  row_dir=a.dataset/"rows"/f"episode{key[0]}_start{key[1]:05d}";npz=row_dir/"temporal.npz";receipt_path=row_dir/"receipt.json";receipt=json.loads(receipt_path.read_text())
  closure=sha(npz)==receipt["npz_sha256"] and receipt["episode"]==key[0] and receipt["start"]==key[1] and sha(spec["source_hdf5"])==spec["source_hdf5_sha256"] and sha(spec["source_window"])==spec["source_window_sha256"] and sha(spec["v461_endpoint_npz"])==spec["v461_endpoint_npz_sha256"]
  with np.load(npz,allow_pickle=False) as z:
   schema=set(z.files)==KEYS;x={k:np.asarray(z[k]).copy() for k in z.files}
  string_arrays=all(x[k].dtype.kind in "US" for k in ("variants","instruction","context_rgb_sha256","context_state_sha256","executed_prefix_action_sha256","public_factual_target_sha256"))
  exact_schema=schema and x["episode"].shape==x["dataset_seed"].shape==x["start"].shape==() and x["episode"].dtype==x["dataset_seed"].dtype==x["start"].dtype==np.int64 and x["instruction"].shape==() and x["variants"].shape==(6,) and string_arrays and x["prefix_lengths"].shape==(8,) and x["prefix_lengths"].dtype==np.int64 and x["factual_public_frame_rgb_mae"].shape==(8,) and x["factual_public_frame_rgb_mae"].dtype==np.float64 and x["pre_future_context_rgb"].shape==(6,8,256,256,3) and x["pre_future_context_rgb"].dtype==np.uint8 and x["temporal_rgb"].shape==(6,8,256,256,3) and x["temporal_rgb"].dtype==np.uint8 and x["public_factual_target_rgb"].shape==(8,256,256,3) and x["public_factual_target_rgb"].dtype==np.uint8 and x["public_factual_target_sha256"].shape==(8,) and x["context_rgb_sha256"].shape==x["context_state_sha256"].shape==x["executed_prefix_action_sha256"].shape==(6,8) and all(x[k].shape==(6,8,d) and x[k].dtype==dtype for k,d,dtype in (("pre_future_context_state",14,np.float32),("pre_future_context_pose",16,np.float64),("pre_future_context_bottle_position",7,np.float64),("temporal_state",14,np.float32),("temporal_pose",16,np.float64),("temporal_bottle_position",7,np.float64))) and x["future_actions"].shape==(6,8,14) and x["future_actions"].dtype==np.float32 and x["history_actions"].shape==(4,14) and x["history_actions"].dtype==np.float32 and np.array_equal(x["prefix_lengths"],np.arange(1,9)) and list(x["variants"].astype("U"))==list(VARIANTS)
  finite=all(np.isfinite(x[k]).all() for k in ("pre_future_context_state","pre_future_context_pose","pre_future_context_bottle_position","temporal_state","temporal_pose","temporal_bottle_position","history_actions","future_actions"))
  with h5py.File(spec["source_hdf5"],"r") as h5:actions=np.asarray(h5["joint_action/vector"],np.float32)
  factual_history=actions[key[1]:key[1]+4].copy();factual_future=actions[key[1]+4:key[1]+12].copy()
  with np.load(spec["source_window"],allow_pickle=False) as z:window_history=np.asarray(z["history_actions"]);window_future=np.asarray(z["future_actions"]);window_target=np.asarray(z["target_frames"])
  window_schema=window_history.shape==(4,14) and window_history.dtype==np.float32 and window_future.shape==(8,14) and window_future.dtype==np.float32 and window_target.shape==(8,256,256,3) and window_target.dtype==np.uint8
  identity=window_schema and int(x["episode"])==key[0] and int(x["start"])==key[1] and int(x["dataset_seed"])==int(spec["dataset_seed"]) and str(x["instruction"])==str(spec["instruction"]) and np.array_equal(x["history_actions"],factual_history) and np.array_equal(x["future_actions"][0],factual_future) and np.array_equal(window_history,factual_history) and np.array_equal(window_future,factual_future) and np.array_equal(x["public_factual_target_rgb"],window_target)
  formula=branches(x["history_actions"],x["future_actions"][0]);lower=np.asarray(pre["per_dim_action_lower"],np.float32);upper=np.asarray(pre["per_dim_action_upper"],np.float32);bounds=bool(np.all(x["future_actions"]>=lower) and np.all(x["future_actions"]<=upper))
  formula_ok=identity and bounds and all(np.array_equal(x["future_actions"][i],formula[n]) and arrsha(x["future_actions"][i])==spec["branch_action_sha256"][n] for i,n in enumerate(VARIANTS)) and arrsha(x["history_actions"])==spec["history_action_sha256"]
  prefix_ok=all(str(x["executed_prefix_action_sha256"][i,k-1])==arrsha(x["future_actions"][i,:k]) for i in range(6) for k in range(1,9))
  stored_hashes=x["context_rgb_sha256"].shape==(6,8) and x["context_state_sha256"].shape==(6,8) and all(str(x["context_rgb_sha256"][i,k])==arrsha(x["pre_future_context_rgb"][i,k]) and str(x["context_state_sha256"][i,k])==arrsha(x["pre_future_context_state"][i,k]) for i in range(6) for k in range(8))
  flat=lambda name:x[name].reshape(48,*x[name].shape[2:])
  same_context=all(np.array_equal(flat(name)[0],flat(name)[i]) for name in ("pre_future_context_rgb","pre_future_context_state","pre_future_context_pose","pre_future_context_bottle_position") for i in range(1,48))
  with np.load(spec["v461_endpoint_npz"],allow_pickle=False) as z:
   oracle={k:np.asarray(z[k]).copy() for k in ("branch_context_rgb","branch_context_state","branch_context_pose","branch_context_bottle_position","endpoint_rgb","endpoint_state","endpoint_bottle_position")}
  oracle_ok=equal_context(x["pre_future_context_rgb"],oracle["branch_context_rgb"]) and equal_context(x["pre_future_context_state"],oracle["branch_context_state"]) and equal_context(x["pre_future_context_pose"],oracle["branch_context_pose"]) and equal_context(x["pre_future_context_bottle_position"],oracle["branch_context_bottle_position"]) and np.array_equal(x["temporal_rgb"][:,-1],oracle["endpoint_rgb"]) and np.array_equal(x["temporal_state"][:,-1],oracle["endpoint_state"]) and np.array_equal(x["temporal_bottle_position"][:,-1],oracle["endpoint_bottle_position"])
  duplicate=np.array_equal(x["temporal_rgb"][0],x["temporal_rgb"][5]) and np.array_equal(x["temporal_state"][0],x["temporal_state"][5]) and np.array_equal(x["temporal_pose"][0],x["temporal_pose"][5]) and np.array_equal(x["temporal_bottle_position"][0],x["temporal_bottle_position"][5])
  effects={}
  for i,n in enumerate(TRANSPORT,1):
   count=sum(float(np.abs(x["temporal_rgb"][i,k].astype(np.float32)-x["temporal_rgb"][0,k].astype(np.float32)).mean())>=1. and (float(np.linalg.norm(x["temporal_state"][i,k]-x["temporal_state"][0,k]))>=.01 or float(np.linalg.norm(x["temporal_bottle_position"][i,k]-x["temporal_bottle_position"][0,k]))>=.005) for k in range(8));effects[n]=count>=2;named[n]+=int(count>=2)
  public_mae=np.asarray([np.abs(x["temporal_rgb"][0,k].astype(np.float32)-x["public_factual_target_rgb"][k].astype(np.float32)).mean() for k in range(8)],np.float64);public_diag=np.array_equal(public_mae,x["factual_public_frame_rgb_mae"]) and all(str(x["public_factual_target_sha256"][k])==arrsha(x["public_factual_target_rgb"][k]) for k in range(8))
  receipt_semantics=bool(receipt.get("context_48way_bitexact"))==same_context and bool(receipt.get("context_v461_oracle_bitexact"))==oracle_ok and bool(receipt.get("k8_v461_rgb_state_bottle_bitexact"))==oracle_ok and bool(receipt.get("executed_prefix_actions_exact"))==prefix_ok and bool(receipt.get("action_bounds_passed"))==bounds and bool(receipt.get("factual_duplicate_temporal_rgb_state_pose_bottle_bitexact"))==duplicate and receipt.get("effective_transport_count")==sum(effects.values()) and bool(receipt.get("effective_transport_gate_passed"))==(sum(effects.values())>=3) and receipt.get("all_fixed_rows_branches_retained") is True
  passed=all((closure,exact_schema,finite,formula_ok,prefix_ok,stored_hashes,same_context,oracle_ok,duplicate,sum(effects.values())>=3,public_diag,receipt_semantics));row_checks.append(passed);rows.append({"episode":key[0],"start":key[1],"npz_sha256":sha(npz),"passed":passed,"effects":effects,"public_factual_frame_rgb_mae_diagnostic":public_mae.tolist()})
 output_bytes=sum(p.stat().st_size for p in a.dataset.rglob("*") if p.is_file());report_rows={(int(x["episode"]),int(x["start"])):x for x in report.get("rows",[])}
 checks={"input_closure":pre_ok and source_guard,"exact_rows_order":list(specs)==expected_order and [(x["episode"],x["start"]) for x in rows]==expected_order and set(report_rows)==set(specs),"row_formula_schema_oracles":all(row_checks),"report_npz_receipt_closure":all(report_rows[k]["npz_sha256"]==row["npz_sha256"] for k,row in zip(expected_order,rows)),"technical_effect_whole_pilot_gate":all(sum(x["effects"].values())>=3 for x in rows) and all(named[n]>=3 for n in TRANSPORT),"resource_gates":report["wall_seconds"]<=1800 and report["gpu_peak_mib"]<=24576 and output_bytes<=268435456,"no_task_outcome_selection":report.get("guards",{}).get("task_reward_success_outcome_selection") is False and report.get("guards",{}).get("all_fixed_rows_branches_frames_retained") is True,"promotion_authorization_boundary":report.get("guards",{}).get("full200_temporal_design_authorized_if_pass") is True and report.get("guards",{}).get("full200_collection_authorized") is False,"no_extra_rows":set(p.name for p in (a.dataset/"rows").iterdir())=={f"episode{e}_start{s:05d}" for e,s in expected_order}}
 passed=all(checks.values());result={"format":AUDIT_FORMAT,"passed":passed,"checks":checks,"rows":rows,"transport_effect_context_counts":named,"evidence_sha256":{"preregistration":sha(a.preregistration),"collector":sha(a.collector),"generation_report":sha(a.generation_report)},"authorization":{"full200_temporal_design_authorized":passed,"full200_collection_authorized":False,"parent_training_authorized":False,"s1_authorized":False,"zero_update_authorized":False,"policy_updates":0,"rl_authorized":False,"real_submission":False}}
 atomic(a.output,result);print(json.dumps({"passed":passed,"checks":checks,"transport_effect_context_counts":named},sort_keys=True));return 0 if passed else 2
if __name__=="__main__":raise SystemExit(main())
