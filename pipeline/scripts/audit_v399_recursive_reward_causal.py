#!/usr/bin/env python3
from __future__ import annotations
import json,sys,tempfile
from pathlib import Path
import audit_v385_recursive_reward_causal as base
from wam_pipeline.v399_posterior_blend_clean_reanchor_runtime import Track2V399PosteriorBlendCleanReanchor
def main()->int:
 argv=list(sys.argv);pi=argv.index('--preregistration')+1;oi=argv.index('--output')+1;source=Path(argv[pi]);output=Path(argv[oi]);payload=json.loads(source.read_text())
 if payload.get('format')!='strict-track2-v399-posterior-blend-clean-reanchor-preregistration-v1':raise RuntimeError('wrong v399 preregistration')
 payload['format']='strict-track2-v385-native-batch-clean-reanchor-preregistration-v1'
 with tempfile.NamedTemporaryFile('w',suffix='.json',delete=False) as f:json.dump(payload,f);temporary=f.name
 argv[pi]=temporary;sys.argv=argv;base.Track2V385NativeBatchCleanReanchor=Track2V399PosteriorBlendCleanReanchor
 try:code=base.main()
 finally:Path(temporary).unlink(missing_ok=True)
 report=json.loads(output.read_text());report['format']='strict-track2-v399-recursive-reward-causal-gate-v1';report['candidate']=payload['model_version'];output.write_text(json.dumps(report,indent=2)+'\n');return code
if __name__=='__main__':raise SystemExit(main())
