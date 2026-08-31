"""Reconcile the sole v469 legacy comparator bug without changing candidate status."""
from __future__ import annotations
import argparse,hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
FORMAT="strict-track2-v470-v469-earlystop-reconciliation-v1"
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def atomic(path,v):
 path=Path(path);tmp=path.with_name(path.name+".tmp")
 if path.exists() or tmp.exists():raise RuntimeError("v470 overwrite")
 with tmp.open("w") as f:json.dump(v,f,indent=2,sort_keys=True);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def main():
 p=argparse.ArgumentParser()
 for n in ("preregistration","legacy-recomputed","output"):p.add_argument(f"--{n}",type=Path,required=True)
 a=p.parse_args();pre=json.loads(a.preregistration.read_text());root=Path(pre["v469_root"]);legacy_original=json.loads((root/"audit_receipt.json").read_text());legacy=json.loads(a.legacy_recomputed.read_text());report=json.loads((root/"result/s0_report.json").read_text());checks={}
 checks["preregistration_source"]=pre.get("format")=="strict-track2-v470-v469-earlystop-reconciliation-preregistration-v1" and sha(Path(__file__))==pre["source"]["reconciler_sha256"]
 checks["immutable_files"]=all((root/rel).is_file() and sha(root/rel)==digest for rel,digest in pre["immutable_sha256"].items()) and not (root/"result/all200_library.npz").exists() and not any("s1" in path.name.lower() for path in root.rglob("*"))
 checks["legacy_full_recompute"]=legacy["checks"]==legacy_original["checks"] and legacy["execution_valid"] is False and legacy["candidate_passed"] is False and legacy["endpoint_parent_data_authorized"] is False and legacy["rl_authorized"] is False and legacy["s0_report_sha256"]==pre["immutable_sha256"]["result/s0_report.json"]
 old_false=[k for k,v in legacy["checks"].items() if not v];checks["legacy_unique_failure"]=old_false==["early_stop_exact"]
 completed=int(report["completed_folds"]);statuses=[bool(x["passed"]) for x in report["folds"]];failed=sum(not x for x in statuses);current_reachable=sum(statuses)+(5-completed)>=4;previous=statuses[:-1];previous_reachable=sum(previous)+(5-len(previous))>=4;all_earlier_reachable=all(sum(statuses[:n])+(5-n)>=4 for n in range(completed));generalized=report["early_stop_mathematically_unreachable"] is True and report["passed"] is False and report["aggregate"] is None and report["all200_library_built"] is False and completed==len(statuses) and failed==2 and statuses[-1] is False and sum(not x for x in previous)==1 and not current_reachable and previous_reachable and all_earlier_reachable
 checks["generalized_first_unreachable_prefix"]=bool(generalized)
 corrected=dict(legacy["checks"]);corrected["early_stop_exact"]=bool(generalized);diff=[k for k in corrected if corrected[k]!=legacy["checks"][k]];checks["only_semantic_diff"]=diff==["early_stop_exact"] and legacy["checks"]["early_stop_exact"] is False and corrected["early_stop_exact"] is True and all(corrected.values())
 execution_valid=all(checks.values());receipt={"format":FORMAT,"created_at":datetime.now(timezone.utc).isoformat(),"execution_valid":execution_valid,"candidate_passed":False,"corrected_legacy_checks":corrected,"reconciliation_checks":checks,"semantic_diff":diff,"v469_s0_report_sha256":sha(root/"result/s0_report.json"),"v469_cache_sha256":sha(root/"result/v169_endpoint_cache.npz"),"all200_library_built":False,"endpoint_parent_authorized":False,"s1_authorized":False,"policy_updates":0,"rl_authorized":False}
 atomic(a.output,receipt);print(json.dumps(receipt,sort_keys=True));return 0 if execution_valid else 3
if __name__=="__main__":raise SystemExit(main())
