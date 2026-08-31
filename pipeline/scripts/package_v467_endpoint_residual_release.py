#!/usr/bin/env python3
"""Atomically package a passed v467 A+D all200 parent."""
from __future__ import annotations
import argparse,hashlib,json,os,shutil
from datetime import datetime,timezone
from pathlib import Path
import torch
FORMAT="track2-v467-canonical-notransport-paired-delta-parent-release-v1";CHECKPOINT_FORMAT="strict-track2-v467-canonical-notransport-paired-delta-checkpoint-v1"
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def copy_sync(source,target):shutil.copy2(source,target);fd=os.open(target,os.O_RDONLY);os.fsync(fd);os.close(fd)
def main():
 p=argparse.ArgumentParser()
 for name in ("preregistration","training-dir","runtime","trainer","contract","v169-release","v169-library","output"):p.add_argument(f"--{name}",type=Path,required=True)
 a=p.parse_args();pre=json.loads(a.preregistration.read_text());report_path=a.training_dir/"training_report.json";s0_path=a.training_dir/"s0_report.json";report=json.loads(report_path.read_text());s0=json.loads(s0_path.read_text());apath=a.training_dir/"all200_appearance_step50.pt";dpath=a.training_dir/"all200_dynamics_action_step50.pt"
 if pre.get("format")!="strict-track2-v467-canonical-notransport-paired-delta-preregistration-v1" or s0.get("passed") is not True or report.get("passed") is not True or report.get("all200_training_performed") is not True:raise RuntimeError("v467 package blocked")
 if sha(a.contract)!=pre["source"]["contract_sha256"] or sha(a.runtime)!=pre["source"]["runtime_sha256"] or sha(a.trainer)!=pre["source"]["trainer_sha256"] or sha(Path(__file__))!=pre["source"]["packager_sha256"]:raise RuntimeError("v467 source drift")
 if report["appearance_checkpoint_sha256"]!=sha(apath) or report["dynamics_checkpoint_sha256"]!=sha(dpath):raise RuntimeError("v467 checkpoint receipt drift")
 closure=hashlib.sha256(json.dumps({"v169":pre["v169"],"source":pre["source"],"v461_files":pre["v461"]["files"]},sort_keys=True,separators=(",",":")).encode()).hexdigest();ast=torch.load(apath,map_location="cpu",weights_only=False);dst=torch.load(dpath,map_location="cpu",weights_only=False)
 for state,scope,schedule in ((ast,"all200-appearance",pre["all200_appearance_schedule_sha256"]),(dst,"all200-dynamics-action",pre["all200_dynamics_schedule_sha256"])):
  if state.get("format")!=CHECKPOINT_FORMAT or state.get("training_scope")!=scope or state.get("step")!=50 or state.get("precision")!="bf16" or state.get("closure_digest")!=closure or state.get("schedule_sha256")!=schedule or state.get("preregistration_sha256")!=sha(a.preregistration):raise RuntimeError("v467 checkpoint closure")
 out=a.output.resolve();partial=out.with_name(out.name+".partial")
 if out.exists() or partial.exists():raise RuntimeError("refuse v467 release overwrite")
 partial.mkdir(parents=True)
 for source,name in ((apath,"all200_appearance_step50.pt"),(dpath,"all200_dynamics_action_step50.pt"),(a.runtime,"v467_canonical_notransport_paired_delta_runtime.py"),(a.preregistration,"preregistration.json"),(s0_path,"s0_report.json"),(report_path,"training_report.json"),(a.contract,"v467_contract.json")):copy_sync(source,partial/name)
 library_manifest=Path(pre["v169"]["library_manifest"])
 manifest={"format":FORMAT,"created_at":datetime.now(timezone.utc).isoformat(),"appearance_checkpoint":"all200_appearance_step50.pt","dynamics_checkpoint":"all200_dynamics_action_step50.pt","canonical_v169_release":str(a.v169_release.resolve()),"canonical_v169_library":str(a.v169_library.resolve()),"v169_library_manifest":str(library_manifest),"closure_digest":closure,"official_reward_runtime_used":False,"policy_modified":False,"rl_authorized":False,"sha256":{"appearance_checkpoint":sha(partial/"all200_appearance_step50.pt"),"dynamics_checkpoint":sha(partial/"all200_dynamics_action_step50.pt"),"runtime_source":sha(partial/"v467_canonical_notransport_paired_delta_runtime.py"),"v169_release_manifest":pre["v169"]["release_manifest_sha256"],"v169_library_manifest":pre["v169"]["library_manifest_sha256"],"preregistration":sha(partial/"preregistration.json"),"s0_report":sha(partial/"s0_report.json"),"training_report":sha(partial/"training_report.json"),"contract":sha(partial/"v467_contract.json")}}
 path=partial/"v467_canonical_notransport_paired_delta_manifest.json"
 with path.open("x") as f:json.dump(manifest,f,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
 fd=os.open(partial,os.O_RDONLY);os.fsync(fd);os.close(fd);os.replace(partial,out);fd=os.open(out.parent,os.O_RDONLY);os.fsync(fd);os.close(fd);print(out);return 0
if __name__=="__main__":raise SystemExit(main())
