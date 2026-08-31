"""Fail-closed static/result audit for the v469 single-shot KNN4 S0."""
from __future__ import annotations
import argparse,ast,hashlib,importlib.util,json,os
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
FORMAT="strict-track2-v469-closed-form-residual-knn-audit-v1"
BRANCHES=("factual","no_transport","scale_0p4","scale_1p25","reverse_direction_0p4")
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(8<<20),b""):h.update(block)
 return h.hexdigest()
def atomic(path,value):
 path=Path(path);tmp=path.with_name(path.name+".tmp")
 if path.exists() or tmp.exists():raise RuntimeError("v469 audit overwrite")
 with tmp.open("w") as f:json.dump(value,f,indent=2,sort_keys=True);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def main():
 p=argparse.ArgumentParser()
 for n in ("preregistration","contract","probe","selection","dataset","result-dir","output"):p.add_argument(f"--{n}",type=Path,required=True)
 a=p.parse_args();pre=json.loads(a.preregistration.read_text());contract=json.loads(a.contract.read_text());report=json.loads((a.result_dir/"s0_report.json").read_text());checks={}
 checks["source_closure"]=sha(a.contract)==pre["contract"]["sha256"] and sha(a.probe)==pre["source"]["probe_sha256"] and sha(Path(__file__))==pre["source"]["auditor_sha256"]
 checks["contract"]=contract.get("format")=="strict-track2-v469-closed-form-residual-knn-contract-v1" and contract.get("status")=="preregistered_public_train_s0_authorized" and contract["frozen_estimator"]["k"]==4 and contract["frozen_estimator"]["weights"].startswith("uniform")
 tree=ast.parse(a.probe.read_text());names={x.id for x in ast.walk(tree) if isinstance(x,ast.Name)};text=a.probe.read_text().lower();checks["static_boundaries"]="track2v169armroutedruntime" in text and "official_reward" not in text and "check_success" not in text and "branch identity as a runtime feature" not in text and "nearest" in names
 checks["input_closure"]=sha(a.selection)==pre["v461"]["selection_sha256"] and a.dataset.resolve()==Path(pre["v461"]["dataset"]).resolve() and all((a.dataset/x["relative"]).is_file() and sha(a.dataset/x["relative"])==x["sha256"] for x in pre["v461"]["files"])
 cache=a.result_dir/"v169_endpoint_cache.npz"
 with np.load(cache,allow_pickle=False) as z:
  requested=z["requested"].copy();canonical=z["canonical"].copy();seeds=z["seeds"].copy();checks["cache_schema"]=requested.shape==(200,5,256,256,3) and requested.dtype==np.uint8 and canonical.shape==(200,256,256,3) and canonical.dtype==np.uint8 and seeds.shape==(200,)
 spec=importlib.util.spec_from_file_location("v469_frozen_probe",a.probe);probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe);arrsha,load_rows,predict_knn,fold_metrics,stable_seed=probe.arrsha,probe.load_rows,probe.predict_knn,probe.fold_metrics,probe.stable_seed
 rows=load_rows(pre,a.dataset,json.loads(a.selection.read_text()));expected_seeds=np.asarray([stable_seed(x["episode"],x["start"]) for x in rows],np.int64);order=[{"episode":x["episode"],"start":x["start"],"seed":int(expected_seeds[i]),"action_sha":[x["keys"][b][0] for b in range(5)]} for i,x in enumerate(rows)];order_digest=hashlib.sha256(json.dumps(order,sort_keys=True,separators=(",",":")).encode()).hexdigest();closure_digest=hashlib.sha256(json.dumps(pre["v468"]["v169"],sort_keys=True,separators=(",",":")).encode()).hexdigest()
 checks["cache_receipt"]=sha(cache)==report["cache"]["sha256"] and report["cache"]["requested_shape"]==[200,5,256,256,3] and report["cache"]["canonical_shape"]==[200,256,256,3] and report["cache"]["requested_array_sha256"]==arrsha(requested) and report["cache"]["canonical_array_sha256"]==arrsha(canonical) and report["cache"]["seed_array_sha256"]==arrsha(seeds) and np.array_equal(seeds,expected_seeds) and report["cache"]["ordered_request_manifest_sha256"]==order_digest and report["cache"]["v169_closure_digest"]==closure_digest
 Y=np.stack([x["target"] for x in rows]).astype(np.float32);C=Y[:,1]-canonical.astype(np.float32);E=(Y-Y[:,1:2])-(requested.astype(np.float32)-canonical[:,None].astype(np.float32));E[:,1]=0;checks["target_formula"]=report["targets"]["appearance_sha256"]==arrsha(C) and report["targets"]["transport_sha256"]==arrsha(E) and report["targets"]["no_transport_exact_zero"] is True and np.array_equal(E[:,1],np.zeros_like(E[:,1]))
 completed=int(report["completed_folds"]);failed=int(report["failed_folds"]);checks["fold_sequence"]=1<=completed<=5 and [x["fold"] for x in report["folds"]]==list(range(completed)) and failed==sum(not x["passed"] for x in report["folds"])
 def exact_fold(x):
  r=x["ratios"];b=x["branch_ratio"];expected=r["overall"]<=.95 and r["uint8"]<=1 and b["factual"]<=1 and all(b[n]<=.98 for n in BRANCHES[1:]) and r["context_only"]<=.95 and r["phase_shuffle"]<=.97 and x["positive_delta_cosine_fraction"]>=.6 and x["nonzero_prediction_delta_fraction"]>=.95
  return bool(x["passed"])==bool(expected)
 recomputed=[]
 for fold in range(completed):
  hold,out,target,base,_=predict_knn(rows,requested,canonical,pre,fold);value,*_=fold_metrics(fold,hold,out,target,base,rows);recomputed.append(value)
 checks["fold_metrics_recomputed"]=all(json.dumps(x,sort_keys=True,separators=(",",":"))==json.dumps(y,sort_keys=True,separators=(",",":")) for x,y in zip(recomputed,report["folds"])) and all(json.loads((a.result_dir/f"fold{x['fold']}_receipt.json").read_text())==x for x in report["folds"])
 checks["fold_gates_exact"]=all(exact_fold(x) for x in report["folds"])
 checks["early_stop_exact"]=(report["early_stop_mathematically_unreachable"] is True and completed==2 and failed==2 and report["aggregate"] is None and report["passed"] is False) or (report["early_stop_mathematically_unreachable"] is False and completed==5)
 if completed==5:
  sums=[x["metric_sums"] for x in report["folds"]];total=lambda k:sum(x[k] for x in sums);branch_action=[sum(x["branch_action"][b] for x in sums) for b in range(5)];branch_base=[sum(x["branch_baseline"][b] for x in sums) for b in range(5)]
  aggregate={"passing_folds":sum(x["passed"] for x in report["folds"]),"continuous_ratio":total("action")/total("baseline"),"uint8_ratio":total("uint8")/total("baseline"),"branch_ratio":{BRANCHES[b]:branch_action[b]/branch_base[b] for b in range(5)},"action_over_context_only":total("action")/total("context_only"),"action_over_phase_shuffle":total("action")/total("phase_shuffle"),"positive_delta_cosine_fraction":float(np.mean([x["positive_delta_cosine_fraction"] for x in report["folds"]])),"nonzero_prediction_delta_fraction":float(np.mean([x["nonzero_prediction_delta_fraction"] for x in report["folds"]])),"improved_episodes":int(sum(x["episode_wins"] for x in report["folds"]))}
  observed=report["aggregate"];same=observed is not None and observed["passing_folds"]==aggregate["passing_folds"] and observed["improved_episodes"]==aggregate["improved_episodes"] and all(np.isclose(observed[k],aggregate[k],rtol=0,atol=1e-12) for k in ("continuous_ratio","uint8_ratio","action_over_context_only","action_over_phase_shuffle","positive_delta_cosine_fraction","nonzero_prediction_delta_fraction")) and all(np.isclose(observed["branch_ratio"][k],aggregate["branch_ratio"][k],rtol=0,atol=1e-12) for k in BRANCHES);checks["aggregate_recomputed"]=bool(same);b=aggregate["branch_ratio"];aggregate_pass=aggregate["passing_folds"]>=4 and aggregate["continuous_ratio"]<=.95 and aggregate["uint8_ratio"]<=1 and b["factual"]<=1 and all(b[x]<=.98 for x in BRANCHES[1:]) and aggregate["action_over_context_only"]<=.95 and aggregate["action_over_phase_shuffle"]<=.97 and aggregate["positive_delta_cosine_fraction"]>=.6 and aggregate["nonzero_prediction_delta_fraction"]>=.95 and aggregate["improved_episodes"]>=12;checks["aggregate_ten_gates_exact"]=bool(report["passed"])==bool(aggregate_pass)
 else:checks["aggregate_recomputed"]=report["aggregate"] is None;checks["aggregate_ten_gates_exact"]=report["passed"] is False
 library=a.result_dir/"all200_library.npz";checks["promotion_exact"]=library.exists()==(report["passed"] is True and report["all200_library_built"] is True)
 if library.exists():
  with np.load(library,allow_pickle=False) as z:
   keys=sorted(z.files);expected_context=np.stack([x["context_feature"] for x in rows]).astype(np.float32);expected_action=np.stack([x["action_feature"] for x in rows]).astype(np.float32);expected_action_sha=np.asarray([[x["keys"][b][0] for b in range(5)] for x in rows]);expected_context_sha=np.asarray([x["keys"][0][1] for x in rows]);checks["library_schema"]=keys==["action","action_sha","appearance","context","context_sha","transport"] and z["context"].shape==(200,192) and z["context"].dtype==np.float32 and z["action"].shape==(200,5,54) and z["action"].dtype==np.float32 and z["appearance"].shape==(200,256,256,3) and z["appearance"].dtype==np.int16 and z["transport"].shape==(200,5,256,256,3) and z["transport"].dtype==np.int16 and z["action_sha"].shape==(200,5) and z["context_sha"].shape==(200,) and np.array_equal(z["context"],expected_context) and np.array_equal(z["action"],expected_action) and np.array_equal(z["appearance"],C.astype(np.int16)) and np.array_equal(z["transport"],E.astype(np.int16)) and np.array_equal(z["action_sha"],expected_action_sha) and np.array_equal(z["context_sha"],expected_context_sha) and np.array_equal(z["transport"][:,1],np.zeros_like(z["transport"][:,1]))
  checks["library_receipt"]=report["all200_library"] is not None and report["all200_library"]["sha256"]==sha(library) and report["all200_library"]["keys"]==keys
 else:checks["library_schema"]=report["all200_library"] is None;checks["library_receipt"]=report["all200_library"] is None
 checks["guards"]=report["guards"]=={"reward_loaded":False,"policy_updates":0,"rl_authorized":False,"hyperparameter_sweep":False}
 execution_valid=all(checks.values());receipt={"format":FORMAT,"created_at":datetime.now(timezone.utc).isoformat(),"execution_valid":execution_valid,"candidate_passed":report["passed"] is True,"checks":checks,"s0_report_sha256":sha(a.result_dir/"s0_report.json"),"cache_sha256":sha(cache),"endpoint_parent_data_authorized":report["passed"] is True and execution_valid,"rl_authorized":False}
 atomic(a.output,receipt);print(json.dumps(receipt,sort_keys=True));return 0 if execution_valid else 3
if __name__=="__main__":raise SystemExit(main())
