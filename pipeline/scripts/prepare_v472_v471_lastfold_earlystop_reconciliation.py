"""Preregister immutable v471 evidence for the last-fold early-stop reconciliation."""
from __future__ import annotations
import argparse,hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
FORMAT="strict-track2-v472-v471-lastfold-earlystop-preregistration-v1"
V471=Path("/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810/v471_group_context_prototype_knn_seed1616_20260824")
CACHE=Path("/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810/v469_closed_form_residual_knn_seed1616_20260824/result/v169_endpoint_cache.npz")
EXPECTED={"preregistration.json":"52d83d9a10b23c80b8ca6cd012bf96549de872131756125c67fbff7312a9601d","result/s0_report.json":"cb6eff0e2ba2a3a38a5d01188e6ed1cf64cf266cc7e3d5d2c3239a50f7134c78","result/fold0_receipt.json":"229e7e80288546004e4a8a5bc8f19ce31c44587fbc44fd488c9ff5389d73dda0","result/fold1_receipt.json":"6fb2923d6be41f79e95bcc0ac32affa2ebcf699c8b731c59a2d4e06bfe7b7885","result/fold2_receipt.json":"6141e6baec01cbb09f866bc4cf0224df898f0a62e648f01802f13d8130f79177","result/fold3_receipt.json":"ac36f03e3a75c8466fe18f924c45fa52512d44e0841e03b2bbe3d935e21f6a7a","result/fold4_receipt.json":"dac6488696d68ef592ddea095dce7c57cdc629b2796f2dc40ac57c9ce4304c97"}
CACHE_SHA="9e98e8cba934a23bf1307df5f4d2847162ee20af59079efbf67d012cfaa001e7"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def atomic(p,v):
 p=Path(p);tmp=p.with_name(p.name+".tmp")
 if p.exists() or tmp.exists():raise RuntimeError("v472 overwrite")
 p.parent.mkdir(parents=True,exist_ok=True)
 with tmp.open("w") as f:json.dump(v,f,indent=2,sort_keys=True);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def main():
 p=argparse.ArgumentParser()
 for n in ("legacy-probe","legacy-auditor","reconciler","launcher","output"):p.add_argument(f"--{n}",type=Path,required=True)
 a=p.parse_args()
 for rel,d in EXPECTED.items():
  if not (V471/rel).is_file() or sha(V471/rel)!=d:raise RuntimeError(f"v472 immutable drift {rel}")
 if sha(CACHE)!=CACHE_SHA:raise RuntimeError("v472 cache drift")
 pre=json.loads((V471/"preregistration.json").read_text());rep=json.loads((V471/"result/s0_report.json").read_text());statuses=[x["passed"] for x in rep["folds"]]
 if pre.get("format")!="strict-track2-v471-group-context-prototype-knn-preregistration-v1" or rep.get("format")!="strict-track2-v471-group-context-prototype-knn-s0-v1" or statuses!=[True,False,True,True,False] or rep.get("completed_folds")!=5 or rep.get("failed_folds")!=2 or rep.get("early_stop_mathematically_unreachable") is not True or rep.get("aggregate") is not None or rep.get("passed") is not False or rep.get("all200_library_built") is not False or rep.get("all200_library") is not None or (V471/"result/all200_group_library.npz").exists():raise RuntimeError("v472 immutable state")
 if sha(a.legacy_probe)!=pre["source"]["probe_sha256"] or sha(a.legacy_auditor)!=pre["source"]["auditor_sha256"]:raise RuntimeError("v472 source drift")
 src=a.legacy_auditor.read_text()
 if src.count('if completed==5:')!=1 or 'agg=m.aggregate(values,recomputed)' not in src:raise RuntimeError("v472 legacy branch semantics drift")
 value={"format":FORMAT,"created_at":datetime.now(timezone.utc).isoformat(),"v471_root":str(V471),"immutable_sha256":EXPECTED,"cache":{"path":str(CACHE),"sha256":CACHE_SHA},"legacy":{"probe_path":str(a.legacy_probe.resolve()),"probe_sha256":sha(a.legacy_probe),"auditor_path":str(a.legacy_auditor.resolve()),"auditor_sha256":sha(a.legacy_auditor),"failure":"TypeError because completed==5 entered aggregate comparison while immutable report aggregate is null"},"source":{"prepare_sha256":sha(Path(__file__)),"reconciler_path":str(a.reconciler.resolve()),"reconciler_sha256":sha(a.reconciler),"launcher_path":str(a.launcher.resolve()),"launcher_sha256":sha(a.launcher)},"authorized_semantic_diff":"evaluate immutable early-stop branch before non-early completed-five aggregate branch","guards":{"candidate_passed":False,"all200_library_built":False,"s1_authorized":False,"policy_updates":0,"rl_authorized":False}}
 atomic(a.output,value);print(json.dumps({"passed":True,"immutable_files":len(EXPECTED),"statuses":statuses},sort_keys=True))
if __name__=="__main__":raise SystemExit(main())
