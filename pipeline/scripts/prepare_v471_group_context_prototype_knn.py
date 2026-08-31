"""Preregister the frozen CPU-only v471 group-context prototype KNN S0."""
from __future__ import annotations
import argparse,hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
FORMAT="strict-track2-v471-group-context-prototype-knn-preregistration-v1";CONTRACT_SHA="cec9a46deb42949f4f12a5d9f81821ab8eb22cb82cacdb90794ed10c5574978a"
V469=Path("/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810/v469_closed_form_residual_knn_seed1616_20260824")
IMM={"preregistration.json":"f895df09787534cb1f1c67a61e1b1c34e0e18dca325981acea607aa2dfe47863","result/s0_report.json":"f612fece6c7fb9ce1fb15471afc10451854b75f2c284f3c283de950b942c749b","result/v169_endpoint_cache.npz":"9e98e8cba934a23bf1307df5f4d2847162ee20af59079efbf67d012cfaa001e7"}
V470=Path("/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810/v470_v469_earlystop_reconciliation_seed1617_20260824/reconciliation_receipt.json");V470_SHA="b7dc4cad04a626e5f7e0d46183314b0ff76da55c5a4131e7c2aaf82db573dcbb"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def atomic(p,v):
 p=Path(p);tmp=p.with_name(p.name+".tmp")
 if p.exists() or tmp.exists():raise RuntimeError("v471 overwrite")
 p.parent.mkdir(parents=True,exist_ok=True)
 with tmp.open("w") as f:json.dump(v,f,indent=2,sort_keys=True);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def main():
 p=argparse.ArgumentParser()
 for n in ("contract","probe","auditor","launcher","output"):p.add_argument(f"--{n}",type=Path,required=True)
 a=p.parse_args();c=json.loads(a.contract.read_text());old=json.loads((V469/"preregistration.json").read_text());recon=json.loads(V470.read_text())
 if sha(a.contract)!=CONTRACT_SHA or c.get("status")!="preregistered_public_train_s0_authorized":raise RuntimeError("v471 contract drift")
 if any(sha(V469/r)!=d for r,d in IMM.items()) or sha(V470)!=V470_SHA or recon.get("execution_valid") is not True or recon.get("candidate_passed") is not False or recon.get("s1_authorized") is not False or recon.get("rl_authorized") is not False:raise RuntimeError("v471 immutable ancestry drift")
 if Path(old["source"]["probe_path"]).resolve()!=Path(old["source"]["probe_path"]) or sha(old["source"]["probe_path"])!=old["source"]["probe_sha256"]:raise RuntimeError("v469 probe source drift")
 value={"format":FORMAT,"created_at":datetime.now(timezone.utc).isoformat(),"contract":{"path":str(a.contract.resolve()),"sha256":sha(a.contract)},"source":{"prepare_sha256":sha(Path(__file__)),"probe_path":str(a.probe.resolve()),"probe_sha256":sha(a.probe),"auditor_path":str(a.auditor.resolve()),"auditor_sha256":sha(a.auditor),"launcher_path":str(a.launcher.resolve()),"launcher_sha256":sha(a.launcher),"v469_probe_path":old["source"]["probe_path"],"v469_probe_sha256":old["source"]["probe_sha256"]},"v469":{"root":str(V469),"immutable_sha256":IMM,"cache_receipt":json.loads((V469/"result/s0_report.json").read_text())["cache"],"preregistration":old},"v470":{"path":str(V470),"sha256":V470_SHA},"estimator":c["frozen_group_estimator"],"features":c["frozen_features"],"gates":c["unchanged_s0_gates"],"guards":{"reward_loaded":False,"gpu_used":False,"policy_updates":0,"s1_authorized":False,"rl_authorized":False,"hyperparameter_sweep":False}}
 atomic(a.output,value);print(json.dumps({"passed":True,"format":FORMAT,"cache_reused":True},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
