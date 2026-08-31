#!/usr/bin/env python3
"""Sole launcher for the v539 public-S1 plus zero-update gate boundary."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import signal
import sys
from pathlib import Path

SEED=1670
PATHS={
 "contract":"/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/v539_v535_public_s1_zero_update_runtime_package_repair_execution_authority_contract.json",
 "preregistration":"/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/v539_v535_public_s1_zero_update_gate_preregistration.json",
 "manifest":"/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/v539_v535_public_s1_zero_update_gate_manifest.json",
 "evaluator_source":"/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/evaluate_v539_v535_public_s1_zero_update_gate.py",
 "auditor_source":"/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/audit_v539_v535_public_s1_zero_update_gate.py",
 "launcher_source":"/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/launch_v539_v535_public_s1_zero_update_gate.py",
 "authority_receipt":"/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810/v539_v535_public_s1_zero_update_runtime_package_repair_execution_authority_seed1670_20260828/authority_receipt.json",
 "output_root":"/root/v539_v535_public_s1_zero_update_gate_seed1670_20260828",
}

def sha(path:Path)->str:
 d=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""): d.update(b)
 return d.hexdigest()

def exact(path:Path,digest:str):
 if path.is_symlink() or not path.is_file() or sha(path)!=digest: raise RuntimeError(f"source drift:{path}")

def parser():
 p=argparse.ArgumentParser()
 for name in ("preregistration","manifest","contract","authority-receipt","evaluator-source","auditor-source","output-root"): p.add_argument("--"+name,type=Path,required=True)
 for name in ("preregistration-sha","manifest-sha","contract-sha","authority-receipt-sha","evaluator-sha","auditor-sha","launcher-sha"): p.add_argument("--"+name,required=True)
 p.add_argument("--seed",type=int,required=True); p.add_argument("--device",choices=("cuda",),required=True)
 return p

def run(argv=None):
 literal=list(sys.argv[1:] if argv is None else argv); args=parser().parse_args(literal)
 if type(args.seed) is not int or args.seed!=SEED: raise RuntimeError("seed strict int")
 for name in ("preregistration","manifest","contract","authority_receipt","evaluator_source","auditor_source","output_root"):
  if str(getattr(args,name))!=PATHS[name]: raise RuntimeError(f"canonical path:{name}")
 launcher=Path(__file__).resolve()
 if str(launcher)!=PATHS["launcher_source"]: raise RuntimeError("canonical launcher path")
 exact(launcher,args.launcher_sha); exact(args.evaluator_source,args.evaluator_sha); exact(args.auditor_source,args.auditor_sha)
 exact(args.preregistration,args.preregistration_sha); exact(args.manifest,args.manifest_sha); exact(args.contract,args.contract_sha); exact(args.authority_receipt,args.authority_receipt_sha)
 blocked={signal.SIGINT,signal.SIGTERM}; previous=signal.pthread_sigmask(signal.SIG_BLOCK,blocked)
 try:
  spec=importlib.util.spec_from_file_location("v539_public_s1_evaluator",args.evaluator_source); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
  evaluator_argv=["--preregistration",str(args.preregistration),"--preregistration-sha",args.preregistration_sha,"--manifest",str(args.manifest),"--manifest-sha",args.manifest_sha,"--contract",str(args.contract),"--contract-sha",args.contract_sha,"--authority-receipt",str(args.authority_receipt),"--authority-receipt-sha",args.authority_receipt_sha,"--evaluator-sha",args.evaluator_sha,"--auditor-source",str(args.auditor_source),"--auditor-sha",args.auditor_sha,"--launcher-source",str(launcher),"--launcher-sha",args.launcher_sha,"--seed",str(args.seed),"--device",args.device,"--output-root",str(args.output_root)]
  parsed=module.parser().parse_args(evaluator_argv); parsed.literal_argv=evaluator_argv
  result=module.execute(parsed)
 finally:
  signal.pthread_sigmask(signal.SIG_SETMASK,previous)
 print(json.dumps({"passed":result["passed"],"status":result["status"],"public_rows":result["public_rows"]},sort_keys=True)); return result

if __name__=="__main__": run()
