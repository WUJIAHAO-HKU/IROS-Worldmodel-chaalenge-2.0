#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True);p.add_argument('--raw',type=Path,required=True);p.add_argument('--preregistration',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 rows=[]
 for line in a.trace.read_text().splitlines():rows.extend(json.loads(line)['batch'])
 def count(key):return sum(bool(x[key]) for x in rows)
 source=np.array([x['source_probability'] for x in rows],float);action=np.array([x['action_probability'] for x in rows],float)
 sequential={};alive=list(rows)
 for key in ('right','source_ready','post_grasp','probability_ready'):
  alive=[x for x in alive if x[key]];sequential[key]=len(alive)
 alive=[x for x in alive if x['failure_signature'] is None];sequential['no_failure_signature']=len(alive)
 alive=[x for x in alive if x['phase_ready']];sequential['phase_ready']=len(alive)
 raw=json.loads(a.raw.read_text())
 report={'format':'strict-track2-v387-route-trace-analysis-v1','created_at':datetime.now(timezone.utc).isoformat(),'requests':len(rows),'batches':len(a.trace.read_text().splitlines()),'unconditional_counts':{k:count(k) for k in ('right','source_ready','post_grasp','probability_ready','phase_ready','route')},'sequential_survivors':sequential,'failure_signatures':{},'source_probability_quantiles':{str(q):float(np.quantile(source,q)) for q in (0,.1,.25,.5,.75,.9,1)},'action_probability_quantiles':{str(q):float(np.quantile(action,q)) for q in (0,.1,.25,.5,.75,.9,.99,1)},'rollout_metrics':raw['metrics'],'protocol':raw['protocol'],'evidence_sha256':{'trace':sha(a.trace),'raw':sha(a.raw),'preregistration':sha(a.preregistration)},'guards':raw['guards']}
 for x in rows:
  s=x['failure_signature'] or 'none';report['failure_signatures'][s]=report['failure_signatures'].get(s,0)+1
 a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
