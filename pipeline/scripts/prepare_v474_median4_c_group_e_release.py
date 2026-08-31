"""Freeze the v474 runtime/package/static-audit closure."""
from __future__ import annotations
import argparse,hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
FORMAT="strict-track2-v474-median4-c-group-e-release-preregistration-v1";V473=Path("/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810/v473_median4_c_v471_e_seed1616_20260824")
IMM={"preregistration.json":"d547c700c6662d985b45e8dc544d64a2ab0997b15c1dc71fe44dcc3c1c769a54","result/s0_report.json":"cc2b91f413ab647bc4366aa8d04a060f44d37803328339d4e1dc79a56c30bad2","audit_receipt.json":"dd9a1a1dc273418413db97a37b0eefcd1c5ff884ccfa2f91eac179d965b0c17b","result/all200_median4_library.npz":"009274cadb1b98a05b4f1297e6c1b6d666428f69086e69b7c139988f30937478"}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def atomic(p,v):
 p=Path(p);tmp=p.with_name(p.name+".tmp")
 if p.exists() or tmp.exists():raise RuntimeError("v474 overwrite")
 p.parent.mkdir(parents=True,exist_ok=True)
 with tmp.open("w") as f:json.dump(v,f,indent=2,sort_keys=True);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def main():
 p=argparse.ArgumentParser()
 for n in ("runtime","packager","auditor","s1-wrapper","launcher","output"):p.add_argument(f"--{n}",type=Path,required=True)
 a=p.parse_args()
 if any(not (V473/r).is_file() or sha(V473/r)!=d for r,d in IMM.items()):raise RuntimeError("v474 v473 drift")
 vp=json.loads((V473/"preregistration.json").read_text());report=json.loads((V473/"result/s0_report.json").read_text());audit=json.loads((V473/"audit_receipt.json").read_text())
 if report.get("passed") is not True or audit.get("execution_valid") is not True or audit.get("candidate_passed") is not True or audit.get("endpoint_parent_data_authorized") is not False or audit.get("s1_authorized") is not False or audit.get("rl_authorized") is not False:raise RuntimeError("v474 authorization")
 v169=vp["v471"]["preregistration"]["v469"]["preregistration"]["v468"]["v169"];closure=hashlib.sha256(json.dumps(v169,sort_keys=True,separators=(",",":")).encode()).hexdigest()
 value={"format":FORMAT,"created_at":datetime.now(timezone.utc).isoformat(),"v473":{"root":str(V473),"immutable_sha256":IMM},"source":{"prepare_sha256":sha(Path(__file__)),"runtime_path":str(a.runtime.resolve()),"runtime_sha256":sha(a.runtime),"packager_path":str(a.packager.resolve()),"packager_sha256":sha(a.packager),"auditor_path":str(a.auditor.resolve()),"auditor_sha256":sha(a.auditor),"s1_wrapper_path":str(a.s1_wrapper.resolve()),"s1_wrapper_sha256":sha(a.s1_wrapper),"launcher_path":str(a.launcher.resolve()),"launcher_sha256":sha(a.launcher)},"v169":v169,"v169_closure_digest":closure,"gate":{"explicit_right":"contains right arm and not left arm","history":"shape4x14 finite and last gripper<0.5","future":"shape8x14 finite and all gripper<0.5","phase":"strict-postclose"},"features":{"context":"last context RGB only -> 8x8x3 block mean","action":"request history/future right6 endpoint+path only","forbidden":"episode,start,request-id,seed,reward,outcome"},"guards":{"reward_loaded":False,"dev_final_loaded":False,"policy_modified":False,"endpoint_parent_data_authorized":False,"s1_authorized":False,"rl_authorized":False}}
 atomic(a.output,value);print(json.dumps({"passed":True,"format":FORMAT,"v169_closure_digest":closure},sort_keys=True))
if __name__=="__main__":raise SystemExit(main())
