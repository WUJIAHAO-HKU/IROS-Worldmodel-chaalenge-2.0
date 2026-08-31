#!/usr/bin/env python3
"""Reuse the frozen v385 gate implementation with the v388 runtime/schema."""
from __future__ import annotations
import json,sys,tempfile
from pathlib import Path
import audit_v385_recursive_reward_causal as base
from wam_pipeline.v388_no_phase_clean_reanchor_runtime import Track2V388NoPhaseCleanReanchor
def main():
 argv=list(sys.argv);i=argv.index('--preregistration')+1;o=argv.index('--output')+1;source=Path(argv[i]);output=Path(argv[o]);payload=json.loads(source.read_text())
 if payload.get('format')!='strict-track2-v388-no-phase-clean-reanchor-preregistration-v1':raise RuntimeError('wrong v388 preregistration')
 payload['format']='strict-track2-v385-native-batch-clean-reanchor-preregistration-v1'
 with tempfile.NamedTemporaryFile('w',suffix='.json',delete=False) as f:json.dump(payload,f);temporary=f.name
 argv[i]=temporary;sys.argv=argv;base.Track2V385NativeBatchCleanReanchor=Track2V388NoPhaseCleanReanchor
 try:code=base.main()
 finally:Path(temporary).unlink(missing_ok=True)
 report=json.loads(output.read_text());report['format']='strict-track2-v388-recursive-reward-causal-gate-v1';report['candidate']=payload['model_version'];output.write_text(json.dumps(report,indent=2)+'\n')
 return code
if __name__=='__main__':raise SystemExit(main())
