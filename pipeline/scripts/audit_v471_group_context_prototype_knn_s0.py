"""Independent result audit for frozen v471 group-context prototype KNN."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def atomic(p,v):
 p=Path(p);tmp=p.with_name(p.name+".tmp")
 with tmp.open("w") as f:json.dump(v,f,indent=2,sort_keys=True);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def main():
 p=argparse.ArgumentParser()
 for n in ("preregistration","contract","probe","selection","dataset","result-dir","output"):p.add_argument(f"--{n}",type=Path,required=True)
 a=p.parse_args();pre=json.loads(a.preregistration.read_text());report=json.loads((a.result_dir/"s0_report.json").read_text());checks={};spec=importlib.util.spec_from_file_location("v471_probe",a.probe);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);old=pre["v469"]["preregistration"]
 checks["format"]=pre.get("format")=="strict-track2-v471-group-context-prototype-knn-preregistration-v1" and report.get("format")=="strict-track2-v471-group-context-prototype-knn-s0-v1"
 checks["source"]=sha(a.contract)==pre["contract"]["sha256"] and sha(a.probe)==pre["source"]["probe_sha256"] and sha(Path(__file__))==pre["source"]["auditor_sha256"] and Path(m.base.__file__).resolve()==Path(pre["source"]["v469_probe_path"]).resolve() and sha(m.base.__file__)==pre["source"]["v469_probe_sha256"]
 checks["data"]=sha(a.selection)==old["v461"]["selection_sha256"] and a.dataset.resolve()==Path(old["v461"]["dataset"]).resolve() and all(sha(a.dataset/x["relative"])==x["sha256"] for x in old["v461"]["files"])
 rows=m.base.load_rows(old,a.dataset,json.loads(a.selection.read_text()));cache=Path(pre["v469"]["root"])/"result/v169_endpoint_cache.npz"
 with np.load(cache,allow_pickle=False) as z:req=z["requested"].copy();can=z["canonical"].copy();seeds=z["seeds"].copy()
 cr=pre["v469"]["cache_receipt"];order_digest,closure_digest=m.cache_digests(rows,pre);checks["cache"]=sha(cache)==pre["v469"]["immutable_sha256"]["result/v169_endpoint_cache.npz"]==report["reused_cache_sha256"] and req.shape==(200,5,256,256,3) and can.shape==(200,256,256,3) and req.dtype==np.uint8 and can.dtype==np.uint8 and seeds.shape==(200,) and m.base.arrsha(req)==cr["requested_array_sha256"] and m.base.arrsha(can)==cr["canonical_array_sha256"] and m.base.arrsha(seeds)==cr["seed_array_sha256"] and order_digest==cr["ordered_request_manifest_sha256"]==report["ordered_request_manifest_sha256"] and closure_digest==cr["v169_closure_digest"]==report["v169_closure_digest"] and np.array_equal(seeds,np.asarray([m.base.stable_seed(x["episode"],x["start"]) for x in rows],np.int64))
 recomputed=[];values=[]
 for fold in range(report["completed_folds"]):
  hold,pred,target,b=m.predict_group(rows,req,can,pre,fold);r,*v=m.base.fold_metrics(fold,hold,pred,target,b,rows);recomputed.append(r);values.append(v)
 checks["folds"]=report["completed_folds"]==len(recomputed) and report["failed_folds"]==sum(not x["passed"] for x in recomputed) and recomputed==report["folds"] and sorted(x.name for x in a.result_dir.glob("fold*_receipt.json"))==[f"fold{i}_receipt.json" for i in range(len(recomputed))] and all(json.loads((a.result_dir/f"fold{i}_receipt.json").read_text())==x for i,x in enumerate(recomputed));statuses=[x["passed"] for x in recomputed];completed=len(statuses);failed=sum(not x for x in statuses);general=failed==2 and statuses[-1] is False and sum(not x for x in statuses[:-1])==1 and sum(statuses)+(5-completed)<4 and all(sum(statuses[:n])+(5-n)>=4 for n in range(completed));checks["early_stop"]=(report["early_stop_mathematically_unreachable"] is True and general and report["aggregate"] is None) or (report["early_stop_mathematically_unreachable"] is False and completed==5 and failed<2)
 if completed==5:
  agg=m.aggregate(values,recomputed);checks["aggregate"]=all(np.isclose(agg[k],report["aggregate"][k],rtol=0,atol=1e-12) for k in ("continuous_ratio","uint8_ratio","action_over_context_only","action_over_phase_shuffle","positive_delta_cosine_fraction","nonzero_prediction_delta_fraction")) and agg["passing_folds"]==report["aggregate"]["passing_folds"] and agg["improved_episodes"]==report["aggregate"]["improved_episodes"] and all(np.isclose(agg["branch_ratio"][k],report["aggregate"]["branch_ratio"][k],rtol=0,atol=1e-12) for k in m.BRANCHES) and bool(report["passed"])==bool(m.aggregate_pass(agg))
 else:checks["aggregate"]=report["aggregate"] is None and report["passed"] is False
 lib=a.result_dir/"all200_group_library.npz";checks["promotion"]=(report["all200_library_built"] is report["passed"]) and lib.exists()==(report["passed"] is True) and ((report["all200_library"] is not None)==(report["passed"] is True))
 if lib.exists():
  Y=np.stack([x["target"] for x in rows]).astype(np.float32);C=(Y[:,1]-can.astype(np.float32)).astype(np.int16);E=((Y-Y[:,1:2])-(req.astype(np.float32)-can[:,None].astype(np.float32))).astype(np.int16);E[:,1]=0
  with np.load(lib,allow_pickle=False) as z:checks["library"]=sorted(z.files)==["action","action_sha","appearance","context","context_sha","transport"] and np.array_equal(z["context"],np.stack([x["context_feature"] for x in rows]).astype(np.float32)) and np.array_equal(z["action"],np.stack([x["action_feature"] for x in rows]).astype(np.float32)) and np.array_equal(z["appearance"],C) and np.array_equal(z["transport"],E) and np.array_equal(z["action_sha"],np.asarray([[x["keys"][b][0] for b in range(5)] for x in rows])) and np.array_equal(z["context_sha"],np.asarray([x["keys"][0][1] for x in rows]))
  checks["library"]=checks["library"] and Path(report["all200_library"]["path"]).resolve()==lib.resolve() and report["all200_library"]["sha256"]==sha(lib)
 else:checks["library"]=report["all200_library"] is None
 checks["guards"]=report["guards"]=={"reward_loaded":False,"gpu_used":False,"policy_updates":0,"s1_authorized":False,"rl_authorized":False,"hyperparameter_sweep":False};valid=all(checks.values());receipt={"format":"strict-track2-v471-group-context-prototype-knn-audit-v1","created_at":datetime.now(timezone.utc).isoformat(),"execution_valid":valid,"candidate_passed":bool(valid and report["passed"] is True),"checks":checks,"report_sha256":sha(a.result_dir/"s0_report.json"),"cache_reused":True,"s1_authorized":False,"rl_authorized":False}
 atomic(a.output,receipt);print(json.dumps(receipt,sort_keys=True));return 0 if valid else 3
if __name__=="__main__":raise SystemExit(main())
