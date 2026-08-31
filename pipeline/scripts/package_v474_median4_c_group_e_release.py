"""Atomically package the passed v473 library as a v474 runtime release."""
from __future__ import annotations
import argparse,hashlib,json,os,shutil
from datetime import datetime,timezone
from pathlib import Path
FORMAT="track2-v474-median4-c-group-e-parent-release-v1"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def cp(s,t):shutil.copy2(s,t);fd=os.open(t,os.O_RDONLY);os.fsync(fd);os.close(fd)
def main():
 p=argparse.ArgumentParser()
 for n in ("preregistration","runtime","output"):p.add_argument(f"--{n}",type=Path,required=True)
 a=p.parse_args();pre=json.loads(a.preregistration.read_text());v=Path(pre["v473"]["root"]);report=json.loads((v/"result/s0_report.json").read_text());audit=json.loads((v/"audit_receipt.json").read_text())
 if pre.get("format")!="strict-track2-v474-median4-c-group-e-release-preregistration-v1" or sha(a.runtime)!=pre["source"]["runtime_sha256"] or sha(Path(__file__))!=pre["source"]["packager_sha256"] or report.get("passed") is not True or audit.get("execution_valid") is not True or audit.get("candidate_passed") is not True:raise RuntimeError("v474 package blocked")
 if any(sha(v/r)!=d for r,d in pre["v473"]["immutable_sha256"].items()):raise RuntimeError("v474 ancestry")
 out=a.output.resolve();tmp=out.with_name(out.name+".partial");v473_pre=json.loads((v/"preregistration.json").read_text());v473_contract=Path(v473_pre["contract"]["path"])
 if not v473_contract.is_file() or sha(v473_contract)!=v473_pre["contract"]["sha256"]:raise RuntimeError("v474 v473 contract drift")
 if out.exists() or tmp.exists():raise RuntimeError("v474 output exists")
 tmp.mkdir(parents=True);items=((v/"result/all200_median4_library.npz","all200_median4_library.npz"),(a.runtime,"v474_v473_median4_parent_runtime.py"),(a.preregistration,"preregistration.json"),(v/"preregistration.json","v473_preregistration.json"),(v473_contract,"v473_contract.json"),(v/"result/s0_report.json","v473_s0_report.json"),(v/"audit_receipt.json","v473_audit_receipt.json"))
 for s,n in items:cp(s,tmp/n)
 vp=tmp/"v169_closure.json"
 with vp.open("x") as f:json.dump(pre["v169"],f,indent=2,sort_keys=True);f.write("\n");f.flush();os.fsync(f.fileno())
 manifest={"format":FORMAT,"created_at":datetime.now(timezone.utc).isoformat(),"library":"all200_median4_library.npz","runtime":"v474_v473_median4_parent_runtime.py","v169_closure":"v169_closure.json","runtime_source_sha256":sha(tmp/"v474_v473_median4_parent_runtime.py"),"package_source_sha256":sha(Path(__file__)),"all200_library_sha256":sha(tmp/"all200_median4_library.npz"),"v473_contract_sha256":sha(tmp/"v473_contract.json"),"v473_preregistration_sha256":sha(tmp/"v473_preregistration.json"),"v473_s0_report_sha256":sha(tmp/"v473_s0_report.json"),"v473_audit_receipt_sha256":sha(tmp/"v473_audit_receipt.json"),"v169_release_manifest_sha256":pre["v169"]["release_manifest_sha256"],"v169_library_manifest_sha256":pre["v169"]["library_manifest_sha256"],"v169_closure_digest":pre["v169_closure_digest"],"median_definition":"sort-float32-middle-two-times-0.5","endpoint_only":True,"gate":pre["gate"],"features":pre["features"],"official_reward_runtime_used":False,"policy_modified":False,"formal":False,"endpoint_parent_data_authorized":False,"s1_authorized":False,"rl_authorized":False,"sha256":{"library":sha(tmp/"all200_median4_library.npz"),"runtime_source":sha(tmp/"v474_v473_median4_parent_runtime.py"),"v169_closure":sha(vp),"preregistration":sha(tmp/"preregistration.json"),"v473_contract":sha(tmp/"v473_contract.json"),"v473_preregistration":sha(tmp/"v473_preregistration.json"),"v473_s0_report":sha(tmp/"v473_s0_report.json"),"v473_audit_receipt":sha(tmp/"v473_audit_receipt.json")}}
 mp=tmp/"v474_median4_c_group_e_manifest.json"
 with mp.open("x") as f:json.dump(manifest,f,indent=2,sort_keys=True);f.write("\n");f.flush();os.fsync(f.fileno())
 fd=os.open(tmp,os.O_RDONLY);os.fsync(fd);os.close(fd);os.replace(tmp,out);fd=os.open(out.parent,os.O_RDONLY);os.fsync(fd);os.close(fd);print(out)
if __name__=="__main__":raise SystemExit(main())
