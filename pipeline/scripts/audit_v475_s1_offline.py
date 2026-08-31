#!/usr/bin/env python3
"""Independent immutable S1 auditor for the frozen v475 serial-v169 endpoint parent."""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np

INPUT_FORMAT="strict-track2-v475-s1-offline-inputs-v1"
REPORT_FORMAT="strict-track2-v475-s1-offline-gate-v1"
SELECTION_FORMAT="strict-track2-v474-s1-action-only-selection-v1"
PREREG_FORMAT="strict-track2-v475-v474-serial-s1-preregistration-v1"
REQUIRED={"format","independent_v169_frames","same_request_v169_frames","v475_true_frames","v475_action_override_frames","target_frames","left_same_request_scalar_v169_frames","left_v475_frames","independent_v169_reward","v475_true_reward","target_reward","episode","start","phase","gate","no_transport","left_episode"}
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
def mae(a,b):return float(np.abs(a.astype(np.int16)-b.astype(np.int16)).mean(dtype=np.float64))
def ratio(a,b):
 if not np.isfinite(a) or not np.isfinite(b) or b<=0:raise RuntimeError("invalid ratio")
 return float(a/b)
def atomic(path,obj):
 path=Path(path);tmp=path.with_name(path.name+".tmp")
 if path.exists() or tmp.exists():raise FileExistsError(path)
 with tmp.open("x",encoding="utf-8") as f:json.dump(obj,f,sort_keys=True,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def closure_ok(pre,a):
 if pre.get("orchestration_boundary")!={"launcher_is_postregistration_orchestrator":True,"launcher_must_bind_formal_preregistration_sha256":True,"launcher_is_not_part_of_execution_closure":True}:return False
 c=pre.get("execution_closure",{});files={"auditor":Path(__file__).resolve(),"selection":a.selection.resolve(),"selection_receipt":a.selection_receipt.resolve(),"static_audit":a.static_audit.resolve()}
 for key,path in files.items():
  rec=c.get(key,{})
  if Path(rec.get("path","")).resolve()!=path or not path.is_file() or sha(path)!=rec.get("sha256"):return False
 for key in ("materializer","generator","s1_wrapper","release_manifest","reward_checkpoint"):
  rec=c.get(key,{});p=Path(rec.get("path",""))
  if not p.is_file() or sha(p)!=rec.get("sha256"):return False
 for key,fn in (("release_tree",treesha),("t5_model_tree",json_treesha)):
  rec=c.get(key,{});p=Path(rec.get("path",""))
  if not p.is_dir() or fn(p)!=rec.get("sha256"):return False
 return True
def main():
 ap=argparse.ArgumentParser()
 for n in ("inputs","generation-report","preregistration","selection","selection-receipt","static-audit","output"):ap.add_argument("--"+n,required=True,type=Path)
 a=ap.parse_args()
 pre=json.loads(a.preregistration.read_text());sel=json.loads(a.selection.read_text());sr=json.loads(a.selection_receipt.read_text());gen=json.loads(a.generation_report.read_text());static=json.loads(a.static_audit.read_text())
 if pre.get("format")!=PREREG_FORMAT or sel.get("format")!=SELECTION_FORMAT:raise RuntimeError("v475 S1 input format")
 if gen.get("format")!="strict-track2-v475-s1-generation-report-v1" or gen.get("passed") is not True:raise RuntimeError("generation report")
 with np.load(a.inputs,allow_pickle=False) as z:
  if set(z.files)!=REQUIRED:raise RuntimeError(f"S1 NPZ schema drift: {sorted(set(z.files)^REQUIRED)}")
  x={k:np.asarray(z[k]) for k in z.files}
 if str(x["format"].item())!=INPUT_FORMAT:raise RuntimeError("S1 NPZ format")
 right_names=("independent_v169_frames","same_request_v169_frames","v475_true_frames","v475_action_override_frames","target_frames")
 if any(x[k].shape!=(32,32,256,256,3) or x[k].dtype!=np.uint8 for k in right_names):raise RuntimeError("right RGB contract")
 if any(x[k].shape!=(12,8,256,256,3) or x[k].dtype!=np.uint8 for k in ("left_same_request_scalar_v169_frames","left_v475_frames")):raise RuntimeError("left RGB contract")
 if any(x[k].shape!=(32,32) or not np.isfinite(x[k]).all() for k in ("independent_v169_reward","v475_true_reward","target_reward")):raise RuntimeError("reward contract")
 if x["gate"].shape!=(32,4) or x["gate"].dtype!=np.bool_ or x["no_transport"].shape!=(32,4) or x["no_transport"].dtype!=np.bool_:raise RuntimeError("decision array contract")
 phases=x["phase"].astype("U");episodes=x["episode"].astype(np.int64);starts=x["start"].astype(np.int64);gate=x["gate"]
 expected_gate=np.asarray(sel["postclose"]["mask32x4"],np.bool_)
 if expected_gate.shape!=(32,4):raise RuntimeError("selection mask shape")
 independent=x["independent_v169_frames"];local=x["same_request_v169_frames"];true=x["v475_true_frames"];override=x["v475_action_override_frames"];target=x["target_frames"]
 b_first=mae(independent[:,:8],target[:,:8]);c_first=mae(true[:,:8],target[:,:8]);b_all=mae(independent,target);c_all=mae(true,target);o_all=mae(override,target)
 br=x["independent_v169_reward"].astype(np.float64);cr=x["v475_true_reward"].astype(np.float64);tr=x["target_reward"].astype(np.float64)
 metrics={"first8_rgb_mae_ratio":ratio(c_first,b_first),"recursive32_rgb_mae_ratio":ratio(c_all,b_all),"reward_prediction_mae_ratio":ratio(float(np.abs(cr-tr).mean()),float(np.abs(br-tr).mean())),"endpoint_reward_prediction_mae_ratio":ratio(float(np.abs(cr[:,-1]-tr[:,-1]).mean()),float(np.abs(br[:,-1]-tr[:,-1]).mean())),"final_reward_mean_ratio":ratio(float(cr[:,-1].mean()),float(br[:,-1].mean())),"true_over_action_shuffle_target_mae":ratio(c_all,o_all),"postclose_structure_fraction":float(gate.mean())}
 repeated=np.repeat(gate,8,axis=1);enabled_protected=np.repeat(gate,7,axis=1).reshape(32,28)
 # Exact comparisons are performed request-wise to avoid accidentally mixing chunks.
 g0_exact=True;enabled_0_6=True
 for i in range(32):
  for c in range(4):
   sl=slice(c*8,(c+1)*8)
   if not gate[i,c]:g0_exact &= bool(np.array_equal(true[i,sl],local[i,sl]))
   else:enabled_0_6 &= bool(np.array_equal(true[i,c*8:c*8+7],local[i,c*8:c*8+7]))
 del repeated,enabled_protected
 decisions=gen.get("right_decisions",[]);odecisions=gen.get("right_override_decisions",[])
 decision_shape=len(decisions)==32 and all(len(r)==4 for r in decisions) and len(odecisions)==32 and all(len(r)==4 for r in odecisions)
 decision_mask=np.asarray([[bool(d["gate"]) for d in r] for r in decisions],np.bool_) if decision_shape else np.empty((0,0),bool)
 nt_decision_ok=decision_shape and all(bool(decisions[i][c]["no_transport"])==bool(odecisions[i][c]["no_transport"]) and ((not bool(decisions[i][c]["no_transport"])) or (decisions[i][c]["action_prototype_sha256"]==[] and odecisions[i][c]["action_prototype_sha256"]==[])) for i in range(32) for c in range(4))
 nt_pixel_exact=decision_shape and all((not bool(decisions[i][c]["no_transport"])) or np.array_equal(true[i,c*8:(c+1)*8],override[i,c*8:(c+1)*8]) for i in range(32) for c in range(4))
 hashes_ok=all(gen.get("rgb_array_sha256",{}).get(k)==arrsha(x[k]) for k in ("independent_v169_frames","same_request_v169_frames","v475_true_frames","v475_action_override_frames","target_frames","left_same_request_scalar_v169_frames","left_v475_frames"))
 checks={
  "static_release_passed":static.get("passed") is True and all(static.get("checks",{}).values()) and static.get("s1_authorized") is False and static.get("rl_authorized") is False and sha(a.static_audit)=="a1fda80ce1aca8ffe33ac1e4e36167df48e3318c33a3ce64a3bac0d70bb037af",
  "selection_receipt_passed":sr.get("passed") is True and all(sr.get("checks",{}).values()) and sr.get("selection_sha256")==sha(a.selection),
  "immutable_input_hashes":closure_ok(pre,a) and gen.get("output_sha256")==sha(a.inputs) and gen.get("selection_sha256")==sha(a.selection) and gen.get("preregistration_sha256")==sha(a.preregistration) and hashes_ok,
  "right_exact32_phase8_each":len(episodes)==32 and np.array_equal(episodes,np.asarray([r["episode"] for r in sel["right"]],np.int64)) and np.array_equal(starts,np.asarray([r["start"] for r in sel["right"]],np.int64)) and np.array_equal(phases,np.asarray([r["phase"] for r in sel["right"]]).astype("U")) and set(map(int,episodes))=={6,7,18,22} and all(int(np.sum(phases==p))==8 for p in ("early","grasp","postgrasp","endpoint")),
  "left_exact12_public_holdout":len(x["left_episode"])==12 and np.array_equal(x["left_episode"].astype(np.int64),np.asarray([r["episode"] for r in sel["left"]],np.int64)),
  "action_only_mask_exact":np.array_equal(gate,expected_gate) and arrsha(gate)==sel["postclose"]["mask_sha256"],
  "postclose_both_represented":bool(gate.any() and (~gate).any()),
  "left_all8_bitexact":np.array_equal(x["left_v475_frames"],x["left_same_request_scalar_v169_frames"]),
  "right_g0_all8_bitexact":g0_exact,
  "right_enabled_frames0_to6_bitexact":enabled_0_6,
  "decision_mask_and_no_transport_exact":decision_shape and np.array_equal(decision_mask,expected_gate) and nt_decision_ok and nt_pixel_exact,
  "scalar_batch_permutation_exact":all(gen.get("interface_probes",{}).get(k) is True for k in ("scalar_batch_bitexact","batch_permutation_bitexact","mixed_left_bitexact")),
  "reward_loaded_once_after_rgb":gen.get("reward_load_count")==1 and gen.get("guards",{}).get("selection_or_calibration_used_reward") is False,
  "first8_rgb_mae_ratio_le_0p998":metrics["first8_rgb_mae_ratio"]<=.998,
  "recursive32_rgb_mae_ratio_le_1p002":metrics["recursive32_rgb_mae_ratio"]<=1.002,
  "reward_prediction_mae_ratio_le_1":metrics["reward_prediction_mae_ratio"]<=1.0,
  "endpoint_reward_prediction_mae_ratio_le_1":metrics["endpoint_reward_prediction_mae_ratio"]<=1.0,
  "final_reward_mean_ratio_ge_0p95":metrics["final_reward_mean_ratio"]>=.95,
  "true_over_action_shuffle_target_mae_le_0p998":metrics["true_over_action_shuffle_target_mae"]<=.998,
  "postclose_fraction_ge_0p2_le_0p8":.2<=metrics["postclose_structure_fraction"]<=.8,
 }
 passed=all(checks.values());report={"format":REPORT_FORMAT,"execution_valid":True,"candidate_passed":passed,"metrics":metrics,"coverage":{"right_samples":32,"right_requests":128,"left_samples":12,"active":int(gate.sum()),"g0":int((~gate).sum())},"checks":checks,"decision":"authorize exactly one raw Pi0.5 zero-update diagnostic" if passed else "reject v475 before zero-update/RL","evidence_sha256":{"inputs":sha(a.inputs),"generation_report":sha(a.generation_report),"preregistration":sha(a.preregistration),"selection":sha(a.selection),"selection_receipt":sha(a.selection_receipt),"static_audit":sha(a.static_audit)},"authorization":{"zero_update_raw_pi05_authorized":passed,"endpoint_parent_data_authorized":False,"policy_updates":0,"formal_rl_authorized":False,"hidden_or_final_data":False,"real_submission":False},"guards":{"models_loaded_by_auditor":False,"official_reward_loaded_by_auditor":False,"runtime_new_bnt_calls":0,"offline_frozen_bnt_residualization_only":True}}
 atomic(a.output,report);print(json.dumps({"candidate_passed":passed,"metrics":metrics,"checks":checks},sort_keys=True));return 0 if passed else 2
if __name__=="__main__":raise SystemExit(main())
