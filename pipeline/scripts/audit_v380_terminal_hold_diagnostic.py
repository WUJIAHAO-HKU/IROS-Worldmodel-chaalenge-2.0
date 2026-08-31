#!/usr/bin/env python3
"""Fixed reward-free train/test audit for terminal-hold v378 rendering."""
from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import audit_v377_recursive_source_blend_sweep as shared

class TerminalHoldRouter(shared.BlendedRouter):
 def predict_batch(self,contexts,histories,futures,seeds,instructions):
  output,probabilities,routes=super().predict_batch(contexts,histories,futures,seeds,instructions)
  for i in np.flatnonzero(routes):output[i]=np.repeat(output[i,-1:,...],8,axis=0)
  return output,probabilities,routes
def run(args,episodes,instructions):
 original=shared.BlendedRouter
 try:
  shared.BlendedRouter=TerminalHoldRouter
  return shared.run_candidate(args,episodes,instructions,{'threshold':0.05,'alpha':0.75})
 finally:shared.BlendedRouter=original
def compare(c,b):
 v=dict(c);v['rgb_ratio']=c['rgb_mae']/b['rgb_mae'];v['temporal_ratio']=c['temporal_error']/b['temporal_error'];return v
def main():
 p=argparse.ArgumentParser()
 for n in ('baseline-release','v375-release','library','gate','windows','instruction-map','preregistration','output'):p.add_argument(f'--{n}',required=True,type=Path)
 p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=8);args=p.parse_args()
 if args.output.exists():raise FileExistsError(args.output)
 prereg=json.loads(args.preregistration.read_text());instructions=json.loads(args.instruction_map.read_text())['episode_to_instruction'];results={}
 for split,key in (('calibration','calibration_episodes'),('test','episode_disjoint_test_episodes')):
  episodes=prereg[key];base=shared.run_baseline(args,episodes,instructions);cand=compare(run(args,episodes,instructions),base);results[split]={'episodes':episodes,'baseline':base,'candidate':cand};print(split.upper(),json.dumps(results[split]),flush=True)
 checks={}
 for split in results:
  c=results[split]['candidate'];checks.update({f'{split}_teacher_routes_zero':c['teacher_routes']==0,f'{split}_rgb_ratio_le_0p80':c['rgb_ratio']<=.80,f'{split}_temporal_ratio_le_0p90':c['temporal_ratio']<=.90,f'{split}_generated_route_rate_ge_0p50':c['generated_route_rate']>=.50})
 report={'format':'strict-track2-v380-terminal-hold-diagnostic-v1','created_at':datetime.now(timezone.utc).isoformat(),'results':results,'checks':checks,'passed':all(checks.values()),'authorizes_frozen_candidate_packaging':all(checks.values()),'guards':{'public_train_only':True,'reward_or_outcomes_used':False,'hidden_or_final_data':False,'real_submission':False}}
 args.output.write_text(json.dumps(report,indent=2)+'\n');print('FINAL',json.dumps({'checks':checks,'passed':report['passed']},indent=2));return 0 if report['passed'] else 3
if __name__=='__main__':raise SystemExit(main())
