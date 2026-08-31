#!/usr/bin/env python3
"""Analyze only outcome-free v398 request telemetry."""
from __future__ import annotations
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def main()->int:
 p=argparse.ArgumentParser();p.add_argument('--trace',required=True,type=Path);p.add_argument('--preregistration',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 if a.output.exists():raise FileExistsError('refusing overwrite')
 prereg=json.loads(a.preregistration.read_text());rows=[];batches=0
 for line in a.trace.read_text().splitlines():
  if line.strip():payload=json.loads(line);rows.extend(payload['batch']);batches+=1
 eligible=[r for r in rows if r['eligible']];relative=[r for r in eligible if r['relative_action_phase_ready']];old=[r for r in eligible if r['continuous_phase_ready']];union=[r for r in eligible if r['union_ready']];prob=np.asarray([r['relative_action_phase_probability'] for r in eligible],dtype=float)
 gate=prereg['gate'];checks={'requests_exact800':len(rows)==800,'batches_exact100':batches==100,'eligible_minimum':len(eligible)>=gate['minimum_eligible_requests'],'relative_routes_minimum':len(relative)>=gate['minimum_relative_routes'],'union_exceeds_old':len(union)>len(old),'output_equivalent_old_route':all(r['route']==r['continuous_phase_ready'] for r in rows)};passed=all(checks.values())
 report={'format':'strict-track2-v398-relative-action-trace-analysis-v1','created_at':datetime.now(timezone.utc).isoformat(),'passed':passed,'requests':len(rows),'batches':batches,'counts':{'eligible':len(eligible),'old_continuous':len(old),'relative_action':len(relative),'union':len(union),'relative_only':sum(r['relative_action_phase_ready'] and not r['continuous_phase_ready'] for r in eligible)},'relative_probability_quantiles':{str(q):float(np.quantile(prob,q)) for q in (0,.1,.25,.5,.75,.9,.99,1)} if len(prob) else None,'checks':checks,'decision':'authorize_v399_integration_and_512_audit' if passed else 'reject_relative_action_rescue_without_integration','evidence_sha256':{'trace':sha(a.trace),'preregistration':sha(a.preregistration)},'guards':{'rollout_success_or_reward_read':False,'raw_rollout_metrics_read':False,'hidden_or_final_data':False,'real_submission':False}}
 a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return 0 if passed else 2
if __name__=='__main__':raise SystemExit(main())
