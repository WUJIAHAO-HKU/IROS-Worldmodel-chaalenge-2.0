"""Independent audit of frozen v473 median4-C / bitexact-v471-E S0."""
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
 a=p.parse_args();pre=json.loads(a.preregistration.read_text());report=json.loads((a.result_dir/"s0_report.json").read_text());checks={};spec=importlib.util.spec_from_file_location("v473_probe",a.probe);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);old=pre["v471"]["preregistration"];v469=old["v469"]["preregistration"]
 checks["format"]=pre.get("format")=="strict-track2-v473-median4-c-v471-e-preregistration-v1" and report.get("format")=="strict-track2-v473-median4-c-v471-e-s0-v1"
 checks["source"]=sha(a.contract)==pre["contract"]["sha256"] and sha(a.probe)==pre["source"]["probe_sha256"] and sha(Path(__file__))==pre["source"]["auditor_sha256"] and Path(m.ref.__file__).resolve()==Path(pre["source"]["v471_probe_path"]).resolve() and sha(m.ref.__file__)==pre["source"]["v471_probe_sha256"] and Path(m.ref.base.__file__).resolve()==Path(pre["source"]["v469_probe_path"]).resolve() and sha(m.ref.base.__file__)==pre["source"]["v469_probe_sha256"]
 checks["ancestry"]=all((Path(pre["v471"]["root"])/r).is_file() and sha(Path(pre["v471"]["root"])/r)==d for r,d in pre["v471"]["immutable_sha256"].items()) and sha(pre["v472"]["path"])==pre["v472"]["sha256"]
 checks["data"]=sha(a.selection)==v469["v461"]["selection_sha256"] and a.dataset.resolve()==Path(v469["v461"]["dataset"]).resolve() and all((a.dataset/x["relative"]).is_file() and sha(a.dataset/x["relative"])==x["sha256"] for x in v469["v461"]["files"])
 rows=m.ref.base.load_rows(v469,a.dataset,json.loads(a.selection.read_text()));cache=Path(old["v469"]["root"])/"result/v169_endpoint_cache.npz"
 with np.load(cache,allow_pickle=False) as z:req=z["requested"].copy();can=z["canonical"].copy();seeds=z["seeds"].copy()
 cr=old["v469"]["cache_receipt"];order,closure=m.ref.cache_digests(rows,old);checks["cache"]=sha(cache)==report["reused_cache_sha256"]==old["v469"]["immutable_sha256"]["result/v169_endpoint_cache.npz"] and req.shape==(200,5,256,256,3) and can.shape==(200,256,256,3) and req.dtype==np.uint8 and can.dtype==np.uint8 and m.ref.base.arrsha(req)==cr["requested_array_sha256"] and m.ref.base.arrsha(can)==cr["canonical_array_sha256"] and m.ref.base.arrsha(seeds)==cr["seed_array_sha256"] and order==cr["ordered_request_manifest_sha256"]==report["ordered_request_manifest_sha256"] and closure==cr["v169_closure_digest"]==report["v169_closure_digest"] and np.array_equal(seeds,np.asarray([m.ref.base.stable_seed(x["episode"],x["start"]) for x in rows],np.int64))
 recomputed=[];values=[]
 for fold in range(report["completed_folds"]):
  hold,reference,candidate,target,b,ev=m.predict_both(rows,req,can,pre,fold);rr,*_=m.ref.base.fold_metrics(fold,hold,reference,target,b,rows);immutable=json.loads((Path(pre["v471"]["root"])/f"result/fold{fold}_receipt.json").read_text());checks[f"v471_reference_fold{fold}"]=rr==immutable
  r,*v=m.ref.base.fold_metrics(fold,hold,candidate,target,b,rows);r["v471_reference_receipt_sha256"]=sha(Path(pre["v471"]["root"])/f"result/fold{fold}_receipt.json");r["bitexact_e_evidence"]=ev;recomputed.append(r);values.append(v)
 checks["folds"]=report["completed_folds"]==len(recomputed)==len(report["folds"]) and report["failed_folds"]==sum(not r["passed"] for r in recomputed) and sorted(x.name for x in a.result_dir.glob("fold*_receipt.json"))==[f"fold{i}_receipt.json" for i in range(len(recomputed))] and recomputed==report["folds"] and all(json.loads((a.result_dir/f"fold{i}_receipt.json").read_text())==r for i,r in enumerate(recomputed)) and all(r["bitexact_e_evidence"]["candidate_e_equals_reference"] and r["bitexact_e_evidence"]["reference_ehat_sha256"]==r["bitexact_e_evidence"]["candidate_ehat_sha256"] and r["bitexact_e_evidence"]["reference_index_manifest_sha256"]==r["bitexact_e_evidence"]["candidate_index_manifest_sha256"] for r in recomputed)
 statuses=[x["passed"] for x in recomputed];failed=sum(not x for x in statuses);completed=len(statuses);first=None
 for n in range(1,completed+1):
  if sum(statuses[:n])+(5-n)<4:first=n;break
 checks["early_stop"]=(report["early_stop_mathematically_unreachable"] is True and first==completed and failed==2 and report["aggregate"] is None) or (report["early_stop_mathematically_unreachable"] is False and completed==5 and failed<2)
 if completed==5 and not report["early_stop_mathematically_unreachable"]:
  plain=[{k:v for k,v in x.items() if k not in ("v471_reference_receipt_sha256","bitexact_e_evidence")} for x in recomputed];agg=m.ref.aggregate(values,plain);checks["aggregate"]=all(np.isclose(agg[k],report["aggregate"][k],rtol=0,atol=1e-12) for k in ("continuous_ratio","uint8_ratio","action_over_context_only","action_over_phase_shuffle","positive_delta_cosine_fraction","nonzero_prediction_delta_fraction")) and agg["passing_folds"]==report["aggregate"]["passing_folds"] and agg["improved_episodes"]==report["aggregate"]["improved_episodes"] and all(np.isclose(agg["branch_ratio"][k],report["aggregate"]["branch_ratio"][k],rtol=0,atol=1e-12) for k in m.BRANCHES) and bool(report["passed"])==bool(m.ref.aggregate_pass(agg))
 else:checks["aggregate"]=report["aggregate"] is None and report["passed"] is False
 lib=a.result_dir/"all200_median4_library.npz";checks["promotion"]=(report["all200_library_built"] is report["passed"]) and lib.exists()==(report["passed"] is True) and ((report["all200_library"] is not None)==(report["passed"] is True))
 if lib.exists():
  Y=np.stack([x["target"] for x in rows]).astype(np.float32);C=(Y[:,1]-can.astype(np.float32)).astype(np.int16);E=((Y-Y[:,1:2])-(req.astype(np.float32)-can[:,None].astype(np.float32))).astype(np.int16);E[:,1]=0
  with np.load(lib,allow_pickle=False) as z:checks["library"]=sorted(z.files)==["action","action_sha","appearance","context","context_sha","transport"] and np.array_equal(z["context"],np.stack([x["context_feature"] for x in rows]).astype(np.float32)) and np.array_equal(z["action"],np.stack([x["action_feature"] for x in rows]).astype(np.float32)) and np.array_equal(z["appearance"],C) and np.array_equal(z["transport"],E) and np.array_equal(z["action_sha"],np.asarray([[x["keys"][b][0] for b in range(5)] for x in rows])) and np.array_equal(z["context_sha"],np.asarray([x["keys"][0][1] for x in rows]))
  checks["library"]=checks["library"] and Path(report["all200_library"]["path"]).resolve()==lib.resolve() and report["all200_library"]["sha256"]==sha(lib) and report["all200_library"]["median_definition"]=="sort-float32-middle-two-times-0.5"
 else:checks["library"]=report["all200_library"] is None
 checks["guards"]=report["guards"]==pre["guards"];valid=all(checks.values());receipt={"format":"strict-track2-v473-median4-c-v471-e-audit-v1","created_at":datetime.now(timezone.utc).isoformat(),"execution_valid":valid,"candidate_passed":bool(valid and report["passed"] is True),"checks":checks,"report_sha256":sha(a.result_dir/"s0_report.json"),"endpoint_parent_data_authorized":False,"s1_authorized":False,"rl_authorized":False}
 atomic(a.output,receipt);print(json.dumps(receipt,sort_keys=True));return 0 if valid else 3
if __name__=="__main__":raise SystemExit(main())
