"""Independent immutable reconciliation of the v471 last-fold early-stop audit branch."""
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
 if p.exists() or tmp.exists():raise RuntimeError("v472 overwrite")
 with tmp.open("w") as f:json.dump(v,f,indent=2,sort_keys=True);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def main():
 p=argparse.ArgumentParser()
 for n in ("preregistration","selection","dataset","output"):p.add_argument(f"--{n}",type=Path,required=True)
 a=p.parse_args();pre=json.loads(a.preregistration.read_text());root=Path(pre["v471_root"]);report=json.loads((root/"result/s0_report.json").read_text());checks={}
 checks["format"]=pre.get("format")=="strict-track2-v472-v471-lastfold-earlystop-preregistration-v1" and report.get("format")=="strict-track2-v471-group-context-prototype-knn-s0-v1"
 checks["immutable"]=all((root/r).is_file() and sha(root/r)==d for r,d in pre["immutable_sha256"].items()) and Path(pre["cache"]["path"]).is_file() and sha(pre["cache"]["path"])==pre["cache"]["sha256"]
 checks["source"]=sha(Path(__file__))==pre["source"]["reconciler_sha256"] and sha(pre["legacy"]["probe_path"])==pre["legacy"]["probe_sha256"] and sha(pre["legacy"]["auditor_path"])==pre["legacy"]["auditor_sha256"]
 legacy_src=Path(pre["legacy"]["auditor_path"]).read_text();checks["legacy_failure_exact"]=legacy_src.count('if completed==5:')==1 and 'agg=m.aggregate(values,recomputed)' in legacy_src and report["completed_folds"]==5 and report["aggregate"] is None
 spec=importlib.util.spec_from_file_location("v471_probe_reconcile",pre["legacy"]["probe_path"]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);old=json.loads((root/"preregistration.json").read_text());v469=old["v469"]["preregistration"];checks["source"]=checks["source"] and Path(m.base.__file__).resolve()==Path(old["source"]["v469_probe_path"]).resolve() and sha(m.base.__file__)==old["source"]["v469_probe_sha256"]
 checks["data"]=sha(a.selection)==v469["v461"]["selection_sha256"] and a.dataset.resolve()==Path(v469["v461"]["dataset"]).resolve() and all((a.dataset/x["relative"]).is_file() and sha(a.dataset/x["relative"])==x["sha256"] for x in v469["v461"]["files"])
 rows=m.base.load_rows(v469,a.dataset,json.loads(a.selection.read_text()))
 with np.load(pre["cache"]["path"],allow_pickle=False) as z:req=z["requested"].copy();can=z["canonical"].copy();seeds=z["seeds"].copy()
 cr=old["v469"]["cache_receipt"];order,closure=m.cache_digests(rows,old);checks["cache"]=req.shape==(200,5,256,256,3) and can.shape==(200,256,256,3) and req.dtype==np.uint8 and can.dtype==np.uint8 and seeds.shape==(200,) and m.base.arrsha(req)==cr["requested_array_sha256"] and m.base.arrsha(can)==cr["canonical_array_sha256"] and m.base.arrsha(seeds)==cr["seed_array_sha256"] and order==cr["ordered_request_manifest_sha256"]==report["ordered_request_manifest_sha256"] and closure==cr["v169_closure_digest"]==report["v169_closure_digest"] and np.array_equal(seeds,np.asarray([m.base.stable_seed(x["episode"],x["start"]) for x in rows],np.int64))
 recomputed=[]
 for fold in range(5):
  hold,pred,target,b=m.predict_group(rows,req,can,old,fold);r,*_=m.base.fold_metrics(fold,hold,pred,target,b,rows);recomputed.append(r)
 checks["folds"]=recomputed==report["folds"] and all(json.loads((root/f"result/fold{i}_receipt.json").read_text())==r for i,r in enumerate(recomputed))
 statuses=[x["passed"] for x in recomputed];first_unreachable=None
 for n in range(1,len(statuses)+1):
  if sum(statuses[:n])+(5-n)<4:first_unreachable=n;break
 corrected_early=report["early_stop_mathematically_unreachable"] is True and first_unreachable==5 and sum(not x for x in statuses)==2 and all(sum(statuses[:n])+(5-n)>=4 for n in range(1,5)) and report["aggregate"] is None and report["passed"] is False
 checks["corrected_early_stop_exact"]=corrected_early
 checks["promotion_blocked"]=report["all200_library_built"] is False and report["all200_library"] is None and not (root/"result/all200_group_library.npz").exists()
 checks["guards"]=report["guards"]=={"reward_loaded":False,"gpu_used":False,"policy_updates":0,"s1_authorized":False,"rl_authorized":False,"hyperparameter_sweep":False}
 valid=all(checks.values());receipt={"format":"strict-track2-v472-v471-lastfold-earlystop-reconciliation-v1","created_at":datetime.now(timezone.utc).isoformat(),"execution_valid":valid,"candidate_passed":False,"all200_library_built":False,"s1_authorized":False,"policy_updates":0,"rl_authorized":False,"checks":checks,"statuses":statuses,"first_mathematically_unreachable_prefix":first_unreachable,"authorized_semantic_diff":pre["authorized_semantic_diff"],"v471_report_sha256":sha(root/"result/s0_report.json")}
 atomic(a.output,receipt);print(json.dumps(receipt,sort_keys=True));return 0 if valid else 3
if __name__=="__main__":raise SystemExit(main())
