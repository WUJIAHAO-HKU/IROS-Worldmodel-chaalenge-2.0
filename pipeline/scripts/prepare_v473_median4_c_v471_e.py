"""Preregister frozen CPU-only v473 median4-C / bitexact-v471-E S0."""
from __future__ import annotations
import argparse,hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
FORMAT="strict-track2-v473-median4-c-v471-e-preregistration-v1";CONTRACT_SHA="06d7869f1beef1c67afe1a64d0de14d992d7b0993f5f7d3739debd9e685b5803"
V471=Path("/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810/v471_group_context_prototype_knn_seed1616_20260824");V472=Path("/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810/v472_v471_lastfold_earlystop_reconciliation_seed1617_20260824/reconciliation_receipt.json")
IMM={"preregistration.json":"52d83d9a10b23c80b8ca6cd012bf96549de872131756125c67fbff7312a9601d","result/s0_report.json":"cb6eff0e2ba2a3a38a5d01188e6ed1cf64cf266cc7e3d5d2c3239a50f7134c78","result/fold0_receipt.json":"229e7e80288546004e4a8a5bc8f19ce31c44587fbc44fd488c9ff5389d73dda0","result/fold1_receipt.json":"6fb2923d6be41f79e95bcc0ac32affa2ebcf699c8b731c59a2d4e06bfe7b7885","result/fold2_receipt.json":"6141e6baec01cbb09f866bc4cf0224df898f0a62e648f01802f13d8130f79177","result/fold3_receipt.json":"ac36f03e3a75c8466fe18f924c45fa52512d44e0841e03b2bbe3d935e21f6a7a","result/fold4_receipt.json":"dac6488696d68ef592ddea095dce7c57cdc629b2796f2dc40ac57c9ce4304c97"};V472_SHA="b40f7de999d06742d80abde93106e455e296c50a96dd3b85a4aa66b24bb4ba31"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def atomic(p,v):
 p=Path(p);tmp=p.with_name(p.name+".tmp")
 if p.exists() or tmp.exists():raise RuntimeError("v473 overwrite")
 p.parent.mkdir(parents=True,exist_ok=True)
 with tmp.open("w") as f:json.dump(v,f,indent=2,sort_keys=True);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def main():
 p=argparse.ArgumentParser()
 for n in ("contract","probe","auditor","launcher","output"):p.add_argument(f"--{n}",type=Path,required=True)
 a=p.parse_args();c=json.loads(a.contract.read_text());old=json.loads((V471/"preregistration.json").read_text());rec=json.loads(V472.read_text())
 if sha(a.contract)!=CONTRACT_SHA or c.get("status")!="preregistered_public_train_s0_authorized":raise RuntimeError("v473 contract")
 if any(sha(V471/r)!=d for r,d in IMM.items()) or sha(V472)!=V472_SHA or rec.get("execution_valid") is not True or rec.get("candidate_passed") is not False:raise RuntimeError("v473 ancestry")
 if Path(old["source"]["probe_path"]).resolve()!=Path(old["source"]["probe_path"]) or sha(old["source"]["probe_path"])!=old["source"]["probe_sha256"] or Path(old["source"]["v469_probe_path"]).resolve()!=Path(old["source"]["v469_probe_path"]) or sha(old["source"]["v469_probe_path"])!=old["source"]["v469_probe_sha256"]:raise RuntimeError("v473 source ancestry")
 value={"format":FORMAT,"created_at":datetime.now(timezone.utc).isoformat(),"contract":{"path":str(a.contract.resolve()),"sha256":CONTRACT_SHA},"source":{"prepare_sha256":sha(Path(__file__)),"probe_path":str(a.probe.resolve()),"probe_sha256":sha(a.probe),"auditor_path":str(a.auditor.resolve()),"auditor_sha256":sha(a.auditor),"launcher_path":str(a.launcher.resolve()),"launcher_sha256":sha(a.launcher),"v471_probe_path":old["source"]["probe_path"],"v471_probe_sha256":old["source"]["probe_sha256"],"v469_probe_path":old["source"]["v469_probe_path"],"v469_probe_sha256":old["source"]["v469_probe_sha256"]},"v471":{"root":str(V471),"immutable_sha256":IMM,"preregistration":old},"v472":{"path":str(V472),"sha256":V472_SHA},"estimator":c["frozen_c_estimator"],"bitexact_e":c["bitexact_v471_e"],"gates":c["unchanged_s0_gates"],"guards":c["guards"]}
 atomic(a.output,value);print(json.dumps({"passed":True,"format":FORMAT},sort_keys=True))
if __name__=="__main__":raise SystemExit(main())
