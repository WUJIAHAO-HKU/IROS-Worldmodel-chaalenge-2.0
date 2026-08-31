#!/usr/bin/env python3
"""Independent immutable audit of v478's twelve new endpoint oracles."""
from __future__ import annotations
import argparse,ast,hashlib,json,os
from pathlib import Path
import h5py,numpy as np

PRE_FORMAT="strict-track2-v478-endpoint-oracle12-preregistration-v1"
SEL_FORMAT="strict-track2-v478-window-eligible-action-only-selection-v1"
REPORT_FORMAT="strict-track2-v478-endpoint-oracle12-generation-report-v1"
AUDIT_FORMAT="strict-track2-v478-endpoint-oracle12-audit-v1"
VARIANTS=("factual","no_transport","scale_0p4","scale_1p25","reverse_direction_0p4","factual_duplicate")
TRANSPORT=VARIANTS[1:5]
KEYS={"episode","dataset_seed","start","variants","instruction","branch_context_rgb","branch_context_state","branch_context_pose","branch_context_bottle_position","history_actions","future_actions","endpoint_rgb","endpoint_state","endpoint_bottle_position","context_rgb_sha256","context_state_sha256","executed_action_sha256"}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def arrsha(x):return hashlib.sha256(np.ascontiguousarray(x).view(np.uint8)).hexdigest()
def atomic(path,obj):
 path=Path(path);tmp=path.with_name(path.name+".tmp")
 if path.exists() or tmp.exists():raise FileExistsError(path)
 with tmp.open("x") as f:json.dump(obj,f,sort_keys=True,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
 os.replace(tmp,path);fd=os.open(str(path.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
def support_tree(root):
 root=Path(root).resolve();paths=sorted(p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts and p.suffix.lower() in (".py",".yml",".yaml"));items=[[p.relative_to(root).as_posix(),sha(p)] for p in paths]
 return hashlib.sha256(json.dumps(items,sort_keys=True,separators=(",",":")).encode()).hexdigest(),len(items)
def branches(history,future):
 a=history[-1,7:13];d=future[:,7:13]-a;out={}
 for n,k in (("no_transport",0.),("scale_0p4",.4),("scale_1p25",1.25),("reverse_direction_0p4",-.4)):
  x=future.copy();x[:,7:13]=a+k*d;out[n]=x
 return {"factual":future.copy(),**out,"factual_duplicate":future.copy()}
def validate(spec,rowdir,lower,upper):
 checks={};npz=rowdir/"endpoint.npz";rp=rowdir/"receipt.json";rec=json.loads(rp.read_text())
 checks["atomic_tree"]={p.name for p in rowdir.iterdir()}=={"endpoint.npz","receipt.json"} and rec.get("npz_sha256")==sha(npz)
 with h5py.File(spec["source_hdf5"],"r") as h:raw=np.asarray(h["joint_action/vector"])
 checks["source_hdf5"]=sha(spec["source_hdf5"])==spec["source_hdf5_sha256"] and raw.dtype==np.float64 and raw.ndim==2 and raw.shape[1]==14 and np.isfinite(raw).all()
 actions=raw.astype(np.float32);start=int(spec["start"]);history=actions[start:start+4].copy();factual=actions[start+4:start+12].copy();b=branches(history,factual);future=np.stack([b[n] for n in VARIANTS])
 with np.load(spec["source_window"],allow_pickle=False) as wz:
  wh=np.asarray(wz["history_actions"]);wf=np.asarray(wz["future_actions"])
  checks["source_window"]=sha(spec["source_window"])==spec["source_window_sha256"] and wh.dtype==np.float32 and wh.shape==(4,14) and wf.dtype==np.float32 and wf.shape==(8,14) and np.array_equal(wh,history) and np.array_equal(wf,factual)
 endpoint={n:float(np.linalg.norm(b[n][-1,7:13]-b["factual"][-1,7:13])) for n in TRANSPORT};path={n:float(np.linalg.norm(b[n][:,7:13]-b["factual"][:,7:13])) for n in TRANSPORT}
 checks["actions"]=arrsha(history)==spec["history_action_sha256"] and arrsha(factual)==spec["factual_future8_action_sha256"] and all(arrsha(b[n])==spec["branch_action_sha256"][n] for n in VARIANTS) and np.all(future>=np.asarray(lower,np.float32)) and np.all(future<=np.asarray(upper,np.float32))
 checks["action_difference_gate"]=all(endpoint[n]>=.01 and path[n]>=.01 and np.isclose(endpoint[n],float(spec["endpoint_diffs"][n]),rtol=1e-7,atol=0) and np.isclose(path[n],float(spec["path_diffs"][n]),rtol=1e-7,atol=0) for n in TRANSPORT)
 with np.load(npz,allow_pickle=False) as z:
  checks["keys_identity"]=set(z.files)==KEYS and int(z["episode"])==int(spec["episode"]) and int(z["dataset_seed"])==int(spec["dataset_seed"]) and int(z["start"])==start and str(z["instruction"])==str(spec["instruction"]) and list(z["variants"].astype("U"))==list(VARIANTS)
  exact=(("branch_context_rgb",(6,256,256,3),np.uint8),("branch_context_state",(6,14),np.float32),("branch_context_pose",(6,16),np.float64),("branch_context_bottle_position",(6,7),np.float64),("history_actions",(4,14),np.float32),("future_actions",(6,8,14),np.float32),("endpoint_rgb",(6,256,256,3),np.uint8),("endpoint_state",(6,14),np.float32),("endpoint_bottle_position",(6,7),np.float64))
  checks["shape_dtype_finite"]=all(z[k].shape==s and z[k].dtype==d and (d==np.uint8 or np.isfinite(z[k]).all()) for k,s,d in exact)
  checks["stored_actions"]=np.array_equal(z["history_actions"],history) and np.array_equal(z["future_actions"],future) and np.array_equal(z["executed_action_sha256"].astype("U"),np.asarray([arrsha(x) for x in future]))
  checks["scalar_string_hash_schema"]=z["episode"].shape==z["dataset_seed"].shape==z["start"].shape==() and z["episode"].dtype==z["dataset_seed"].dtype==z["start"].dtype==np.int64 and z["instruction"].shape==() and z["instruction"].dtype.kind in "SU" and z["variants"].shape==(6,) and z["variants"].dtype.kind in "SU" and z["context_rgb_sha256"].shape==(6,) and z["context_rgb_sha256"].dtype.kind in "SU" and z["context_state_sha256"].shape==(6,) and z["context_state_sha256"].dtype.kind in "SU" and z["executed_action_sha256"].shape==(6,) and z["executed_action_sha256"].dtype.kind in "SU"
  checks["stored_context_hashes"]=np.array_equal(z["context_rgb_sha256"].astype("U"),np.asarray([arrsha(x) for x in z["branch_context_rgb"]])) and np.array_equal(z["context_state_sha256"].astype("U"),np.asarray([arrsha(x) for x in z["branch_context_state"]]))
  checks["same_context_duplicate"]=all(np.array_equal(z["branch_context_rgb"][0],z["branch_context_rgb"][i]) and np.array_equal(z["branch_context_state"][0],z["branch_context_state"][i]) and np.array_equal(z["branch_context_pose"][0],z["branch_context_pose"][i]) and np.array_equal(z["branch_context_bottle_position"][0],z["branch_context_bottle_position"][i]) for i in range(1,6)) and np.array_equal(z["endpoint_rgb"][0],z["endpoint_rgb"][5]) and np.array_equal(z["endpoint_state"][0],z["endpoint_state"][5]) and np.array_equal(z["endpoint_bottle_position"][0],z["endpoint_bottle_position"][5])
  effects={}
  for i,n in enumerate(TRANSPORT,1):
   rgb=float(np.abs(z["endpoint_rgb"][i].astype(np.float32)-z["endpoint_rgb"][0].astype(np.float32)).mean());state=float(np.linalg.norm(z["endpoint_state"][i]-z["endpoint_state"][0]));bottle=float(np.linalg.norm(z["endpoint_bottle_position"][i]-z["endpoint_bottle_position"][0]));effects[n]={"endpoint_rgb_mae":rgb,"endpoint_qpos_l2":state,"bottle_position_l2":bottle,"passed":rgb>=1 and (state>=.01 or bottle>=.005)}
 checks["receipt_identity_guards"]=rec.get("format")=="strict-track2-v478-endpoint-oracle12-row-v1" and rec.get("episode")==int(spec["episode"]) and rec.get("start")==start and rec.get("source_hdf5_sha256")==spec["source_hdf5_sha256"] and rec.get("source_action_raw_dtype")=="float64" and rec.get("canonical_action_dtype")=="float32" and rec.get("history_action_sha256")==spec["history_action_sha256"] and rec.get("branch_action_sha256")==spec["branch_action_sha256"] and rec.get("effect_used_for_retry_or_filter") is False and rec.get("task_reward_success_outcome_consumed") is False
 checks["receipt_metrics"]=rec.get("effects")==effects and rec.get("context_bitexact") is True and rec.get("executed_actions_exact") is True and rec.get("action_bounds_passed") is True and rec.get("action_diff_passed") is True and all(np.isclose(float(rec.get("action_endpoint_diff_l2",{}).get(n,-1)),endpoint[n],rtol=1e-7,atol=0) for n in TRANSPORT) and rec.get("duplicate_passed") is True
 return all(checks.values()),checks,effects,sha(npz)
def main():
 ap=argparse.ArgumentParser()
 for n in ("preregistration","selection","collector","oracle-root","generation-report","support-root","task-config","resize-source","output"):ap.add_argument("--"+n,type=Path,required=True)
 a=ap.parse_args();pre=json.loads(a.preregistration.read_text());sel=json.loads(a.selection.read_text());report=json.loads(a.generation_report.read_text());closure=pre["execution_closure"]
 master_path=Path(pre["master_contract"]["path"]).resolve();master=json.loads(master_path.read_text());selection_receipt_path=Path(pre["selection"]["receipt_path"]).resolve();selection_receipt=json.loads(selection_receipt_path.read_text());materializer=Path(closure["materializer"]["path"]).resolve()
 ancestry_ok=sha(master_path)==pre["master_contract"]["sha256"] and master.get("format")=="strict-track2-v478-public-train-temporal200-collection-contract-v1" and master.get("status")=="phase_a_endpoint_oracle12_authorized_phase_b_temporal_false" and master.get("phase_b_temporal200",{}).get("authorized") is False and master.get("guards",{}).get("phase_b_temporal_collection_authorized") is False and Path(pre["selection"]["path"]).resolve()==a.selection.resolve() and pre["selection"]["sha256"]==sha(a.selection) and sha(selection_receipt_path)==pre["selection"]["receipt_sha256"] and selection_receipt.get("format")=="strict-track2-v478-window-eligible-selection-receipt-v1" and selection_receipt.get("passed") is True and Path(selection_receipt["selection"]["path"]).resolve()==a.selection.resolve() and selection_receipt["selection"]["sha256"]==sha(a.selection) and Path(selection_receipt["materializer"]["path"]).resolve()==materializer and selection_receipt["materializer"]["sha256"]==sha(materializer)==closure["materializer"]["sha256"]
 source=Path(a.collector).read_text();tree=ast.parse(source);forbidden=sum(isinstance(x,ast.Call) and ((isinstance(x.func,ast.Attribute) and x.func.attr in {"check_success","gen_sparse_reward_data","save_pcd","reward","get_reward"}) or (isinstance(x.func,ast.Name) and x.func.id in {"gen_sparse_reward_data","save_pcd"})) for x in ast.walk(tree));support_digest,support_count=support_tree(a.support_root)
 new={(int(x["episode"]),int(x["start"])) for x in sel["oracle_reuse"]["new_contexts"]};specs=pre["contexts"]
 input_ok=pre.get("format")==PRE_FORMAT and pre.get("status")=="preregistered_public_train_endpoint_oracle_collection_authorized" and sel.get("format")==SEL_FORMAT and sha(a.selection)==closure["selection"]["sha256"] and len(specs)==len(new)==12 and {(int(x["episode"]),int(x["start"])) for x in specs}==new and Path(pre["oracle_root"]).resolve()==a.oracle_root.resolve() and Path(closure["collector"]["path"]).resolve()==a.collector.resolve() and closure["collector"]["sha256"]==sha(a.collector) and Path(closure["generator"]["path"]).resolve()==Path(pre["execution_closure"]["generator"]["path"]).resolve() and report.get("generator_sha256")==closure["generator"]["sha256"]==sha(closure["generator"]["path"]) and Path(closure["auditor"]["path"]).resolve()==Path(__file__).resolve() and closure["auditor"]["sha256"]==sha(Path(__file__).resolve()) and report.get("format")==REPORT_FORMAT and report.get("passed") is True and report.get("preregistration_sha256")==sha(a.preregistration) and report.get("collector_sha256")==sha(a.collector) and Path(closure["support_root"]["path"]).resolve()==a.support_root.resolve() and support_digest==closure["support_root"]["source_config_tree_sha256"]==report.get("support_source_config_tree_sha256") and support_count==closure["support_root"]["source_config_file_count"]==report.get("support_source_config_file_count") and Path(closure["task_config"]["path"]).resolve()==a.task_config.resolve() and sha(a.task_config)==closure["task_config"]["sha256"]==report.get("task_config_sha256") and Path(closure["resize_source"]["path"]).resolve()==a.resize_source.resolve() and sha(a.resize_source)==closure["resize_source"]["sha256"]==report.get("resize_source_sha256") and report.get("runtime_provenance")==pre.get("runtime_provenance_expected") and forbidden==0
 input_ok=input_ok and ancestry_ok
 rowdirs={p.name:p for p in (a.oracle_root/"rows").iterdir()};expected={f"episode{int(x['episode'])}_start{int(x['start']):05d}" for x in specs};row_results=[];counts={n:0 for n in TRANSPORT};report_rows={(int(x["episode"]),int(x["start"])):x["npz_sha256"] for x in report.get("rows",[])}
 for spec in specs:
  name=f"episode{int(spec['episode'])}_start{int(spec['start']):05d}";ok,detail,effect,nsha=validate(spec,rowdirs[name],pre["per_dim_action_lower"],pre["per_dim_action_upper"]);row_results.append({"episode":int(spec["episode"]),"start":int(spec["start"]),"passed":ok,"checks":detail,"npz_sha256":nsha});
  for n in TRANSPORT:counts[n]+=int(effect[n]["passed"])
  input_ok &= report_rows.get((int(spec["episode"]),int(spec["start"])))==nsha
 checks={"input_source_closure":input_ok,"exact12_order":[(x["episode"],x["start"]) for x in row_results]==[(int(x["episode"]),int(x["start"])) for x in specs],"all_rows_independently_valid":all(x["passed"] for x in row_results),"exact_atomic_tree":set(rowdirs)==expected and {p.name for p in a.oracle_root.iterdir()}=={"rows","generation_report.json"},"effect_diagnostic_not_selection":report.get("guards",{}).get("effect_used_for_retry_or_filter") is False,"no_task_outcome_model_policy":report.get("guards",{}).get("training_authorized") is False and report.get("guards",{}).get("s1_authorized") is False and report.get("guards",{}).get("policy_updates")==0 and report.get("guards",{}).get("rl_authorized") is False}
 passed=all(checks.values());result={"format":AUDIT_FORMAT,"passed":passed,"checks":checks,"rows":row_results,"transport_effect_context_counts_diagnostic":counts,"evidence_sha256":{"preregistration":sha(a.preregistration),"selection":sha(a.selection),"collector":sha(a.collector),"generation_report":sha(a.generation_report)},"authorization":{"phase_b_temporal_collection_authorized":passed,"training_authorized":False,"s1_authorized":False,"zero_update_authorized":False,"policy_updates":0,"rl_authorized":False}}
 atomic(a.output,result);return 0 if passed else 3
if __name__=="__main__":raise SystemExit(main())
