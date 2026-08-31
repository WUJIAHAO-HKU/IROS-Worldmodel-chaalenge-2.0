#!/usr/bin/env python3
"""Sole fixed-budget training launcher; exact argv and no budget override."""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path

SEED=1672; OUTPUT=Path('/root/v541_v540_fixed_budget_all200_training_seed1624_20260828')
def parser():
 p=argparse.ArgumentParser()
 for name in ('preregistration','manifest','contract','authority-receipt','trainer-source','auditor-source','output-root'): p.add_argument('--'+name,type=Path,required=True)
 for name in ('preregistration-sha','manifest-sha','contract-sha','authority-receipt-sha','trainer-sha','auditor-sha','launcher-sha'): p.add_argument('--'+name,required=True)
 p.add_argument('--seed',type=int,required=True); p.add_argument('--device',choices=['cuda'],required=True); return p
def exact(path,digest):
 import hashlib,os,stat
 st=os.lstat(path)
 if not stat.S_ISREG(st.st_mode) or stat.S_ISLNK(st.st_mode) or hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest: raise RuntimeError('source '+str(path))
def run(argv=None):
 literal=list(argv or []); args=parser().parse_args(literal)
 if args.seed!=SEED or args.output_root!=OUTPUT: raise RuntimeError('literal args')
 launcher=Path(__file__).resolve()
 for path,digest in ((launcher,args.launcher_sha),(args.preregistration,args.preregistration_sha),(args.manifest,args.manifest_sha),(args.contract,args.contract_sha),(args.authority_receipt,args.authority_receipt_sha),(args.trainer_source,args.trainer_sha),(args.auditor_source,args.auditor_sha)): exact(path,digest)
 spec=importlib.util.spec_from_file_location('v541_fixed_budget_trainer',args.trainer_source); trainer=importlib.util.module_from_spec(spec); spec.loader.exec_module(trainer)
 trainer_argv=['--preregistration',str(args.preregistration),'--preregistration-sha',args.preregistration_sha,'--manifest',str(args.manifest),'--manifest-sha',args.manifest_sha,'--contract',str(args.contract),'--contract-sha',args.contract_sha,'--authority-receipt',str(args.authority_receipt),'--authority-receipt-sha',args.authority_receipt_sha,'--trainer-sha',args.trainer_sha,'--auditor-source',str(args.auditor_source),'--auditor-sha',args.auditor_sha,'--launcher-source',str(launcher),'--launcher-sha',args.launcher_sha,'--seed',str(args.seed),'--device',args.device,'--output-root',str(args.output_root)]
 parsed=trainer.parser().parse_args(trainer_argv); parsed.literal_argv=trainer_argv
 result=trainer.execute(parsed)
 print(json.dumps({'passed':result['passed'],'status':result['status'],'public_steps':result['public_steps']},sort_keys=True)); return result
def main(): run(__import__('sys').argv[1:]); return 0
if __name__=='__main__': raise SystemExit(main())
