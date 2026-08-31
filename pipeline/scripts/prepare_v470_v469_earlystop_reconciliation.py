"""Freeze immutable v469 evidence for the v470 early-stop comparator reconciliation."""
from __future__ import annotations
import argparse,hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
FORMAT="strict-track2-v470-v469-earlystop-reconciliation-preregistration-v1"
EXPECTED={"preregistration.json":"f895df09787534cb1f1c67a61e1b1c34e0e18dca325981acea607aa2dfe47863","result/s0_report.json":"f612fece6c7fb9ce1fb15471afc10451854b75f2c284f3c283de950b942c749b","result/v169_endpoint_cache.npz":"9e98e8cba934a23bf1307df5f4d2847162ee20af59079efbf67d012cfaa001e7","result/fold0_receipt.json":"508df901b9df7bce385b0e1ae7fcf5781e24b34b63d21e0c55def1d2fd8c15ad","result/fold1_receipt.json":"22cbe5455fad3a44a2f0134265c70e616cbad7894bd37849d7fbb56ba4b2a2ee","result/fold2_receipt.json":"2c723370e9acdd77e88177352e2280ecf084ab0d8682012c34069b8062f056c7","audit_receipt.json":"d2928069d06fea866327843de7c626620c3b26afd20ae33ef7989c0d1b265525"}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def atomic(path,v):
 path=Path(path);tmp=path.with_name(path.name+".tmp")
 if path.exists() or tmp.exists():raise RuntimeError("v470 overwrite")
 path.parent.mkdir(parents=True,exist_ok=True)
 with tmp.open("w") as f:json.dump(v,f,indent=2,sort_keys=True);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def main():
 p=argparse.ArgumentParser()
 for n in ("v469-reg","legacy-auditor","reconciler","launcher","output"):p.add_argument(f"--{n}",type=Path,required=True)
 a=p.parse_args();root=a.v469_reg.resolve()
 for rel,digest in EXPECTED.items():
  path=root/rel
  if not path.is_file() or sha(path)!=digest:raise RuntimeError(f"v470 immutable drift {rel}")
 pre=json.loads((root/"preregistration.json").read_text());report=json.loads((root/"result/s0_report.json").read_text());legacy=json.loads((root/"audit_receipt.json").read_text())
 if pre.get("format")!="strict-track2-v469-closed-form-residual-knn-preregistration-v1" or report.get("format")!="strict-track2-v469-closed-form-residual-knn-s0-v1" or report.get("passed") is not False or report.get("all200_library_built") is not False or (root/"result/all200_library.npz").exists():raise RuntimeError("v470 v469 state drift")
 false=[k for k,v in legacy["checks"].items() if not v]
 if false!=["early_stop_exact"] or legacy.get("candidate_passed") is not False or legacy.get("execution_valid") is not False or legacy.get("endpoint_parent_data_authorized") is not False or legacy.get("rl_authorized") is not False:raise RuntimeError("v470 legacy failure not unique")
 if sha(a.legacy_auditor)!=pre["source"]["auditor_sha256"]:raise RuntimeError("v470 legacy auditor source drift")
 value={"format":FORMAT,"created_at":datetime.now(timezone.utc).isoformat(),"v469_root":str(root),"immutable_sha256":EXPECTED,"v469_source":{"probe_path":pre["source"]["probe_path"],"probe_sha256":pre["source"]["probe_sha256"],"auditor_path":str(a.legacy_auditor.resolve()),"auditor_sha256":sha(a.legacy_auditor),"contract_path":pre["contract"]["path"],"contract_sha256":pre["contract"]["sha256"],"selection_path":pre["v461"]["selection_path"],"dataset":pre["v461"]["dataset"]},"source":{"prepare_sha256":sha(Path(__file__)),"reconciler_path":str(a.reconciler.resolve()),"reconciler_sha256":sha(a.reconciler),"launcher_path":str(a.launcher.resolve()),"launcher_sha256":sha(a.launcher)},"authorized_semantic_diff":"generalize early_stop_exact from completed==2 to first prefix where cumulative failures reaches two and four-of-five becomes unreachable","guards":{"candidate_passed":False,"all200_library_built":False,"s1_authorized":False,"policy_updates":0,"rl_authorized":False}}
 atomic(a.output,value);print(json.dumps({"passed":True,"format":FORMAT,"immutable_files":len(EXPECTED)},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
