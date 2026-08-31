#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def main()->int:
 p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True);p.add_argument('--raw',type=Path,required=True);p.add_argument('--preregistration',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError('refusing overwrite')
 prereg=json.loads(a.preregistration.read_text());lines=a.trace.read_text().splitlines();rows=[x for line in lines for x in json.loads(line)['batch']];raw=json.loads(a.raw.read_text());metrics=raw['metrics'];trajectories=int(round(metrics['num_trajectories']));successes=int(round(float(metrics['success_once'])*trajectories));alive=rows
 sequential={}
 for key in ('right','source_ready','post_grasp','probability_ready'):alive=[r for r in alive if r[key]];sequential[key]=len(alive)
 alive=[r for r in alive if r['failure_signature'] is None];sequential['no_failure_signature']=len(alive)
 for key in ('hard_phase_ready','continuous_phase_ready','supported_phase_ready','route'):sequential[key]=sum(bool(r[key]) for r in alive)
 values=np.asarray([r['continuous_phase_probability'] for r in alive if r['continuous_phase_probability'] is not None],float);protocol=bool(raw['protocol']['policy_updates']==0 and raw['protocol']['checkpoint_writes']==0 and raw['protocol']['rollout_epochs']==1 and trajectories==32 and len(rows)==800);gate=prereg['gate'];passed=bool(protocol and successes>=gate['minimum_successes'] and sequential['supported_phase_ready']>=gate['minimum_supported_routes'])
 report={'format':'strict-track2-v401-v400-route-trace-analysis-v1','created_at':datetime.now(timezone.utc).isoformat(),'passed':passed,'successes':successes,'trajectories':trajectories,'success_fraction':f'{successes}/{trajectories}','success_rate':float(metrics['success_once']),'minimum_successes':gate['minimum_successes'],'requests':len(rows),'batches':len(lines),'sequential_survivors':sequential,'phase_probability_quantiles':{str(q):float(np.quantile(values,q)) for q in (0,.1,.25,.5,.75,.9,.99,1)},'protocol_ok':protocol,'decision':'permit_one_conservative_rl_update' if passed else 'reject_v400_before_rl','rollout_metrics':metrics,'evidence_sha256':{'trace':sha(a.trace),'raw':sha(a.raw),'preregistration':sha(a.preregistration)},'guards':raw['guards']};a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
