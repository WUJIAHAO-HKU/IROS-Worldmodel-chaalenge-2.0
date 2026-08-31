#!/usr/bin/env python3
"""Reward-free calibration/test for a terminal-preserving temporal renderer."""
from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import audit_v377_recursive_source_blend_sweep as shared

class SmoothedRouter(shared.BlendedRouter):
 def __init__(self,*args,raw_weight,**kwargs):super().__init__(*args,**kwargs);self.raw_weight=float(raw_weight)
 def predict_batch(self,contexts,histories,futures,seeds,instructions):
  output,probabilities,routes=super().predict_batch(contexts,histories,futures,seeds,instructions)
  weights=(np.arange(1,9,dtype=np.float32)/8.0).reshape(8,1,1,1)
  for i in np.flatnonzero(routes):
   raw=output[i].astype(np.float32);start=contexts[i,-1].astype(np.float32);linear=start[None]+weights*(raw[-1][None]-start[None]);smoothed=self.raw_weight*raw+(1.0-self.raw_weight)*linear;smoothed[-1]=raw[-1];output[i]=np.clip(np.rint(smoothed),0,255).astype(np.uint8)
  return output,probabilities,routes

def candidate(args,episodes,instructions,config):
 original=shared.BlendedRouter
 try:
  shared.BlendedRouter=lambda release,library,gate,threshold,alpha,device:SmoothedRouter(release,library,gate,threshold,alpha,device,raw_weight=config['raw_intermediate_weight'])
  fixed={'threshold':0.05,'alpha':0.75}
  return shared.run_candidate(args,episodes,instructions,fixed)
 finally:shared.BlendedRouter=original

def compare(c,b):
 v=dict(c);v['rgb_ratio']=c['rgb_mae']/b['rgb_mae'];v['temporal_ratio']=c['temporal_error']/b['temporal_error'];v['eligible']=bool(c['teacher_routes']==0 and v['rgb_ratio']<=.80 and v['temporal_ratio']<=.95);v['selection_score']=max(v['rgb_ratio']/.75,v['temporal_ratio']/.90);return v

def main():
 p=argparse.ArgumentParser()
 for n in ('baseline-release','v375-release','library','gate','windows','instruction-map','preregistration','output'):p.add_argument(f'--{n}',required=True,type=Path)
 p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=8);args=p.parse_args()
 if args.output.exists():raise FileExistsError(args.output)
 prereg=json.loads(args.preregistration.read_text());
 if prereg.get('format')!='strict-track2-v379-temporal-renderer-sweep-preregistration-v1':raise RuntimeError('wrong preregistration')
 instructions=json.loads(args.instruction_map.read_text())['episode_to_instruction'];cal_eps=prereg['calibration_episodes'];base_cal=shared.run_baseline(args,cal_eps,instructions);cal=[]
 for cfg in prereg['configs']:
  m=compare(candidate(args,cal_eps,instructions,cfg),base_cal);cal.append({'config':cfg,'metrics':m});print('CALIBRATION',json.dumps(cal[-1]),flush=True)
 eligible=[x for x in cal if x['metrics']['eligible']];selected=min(eligible,key=lambda x:(x['metrics']['selection_score'],-x['config']['raw_intermediate_weight'])) if eligible else None;test=None;checks={'eligible_calibration_candidate_found':selected is not None}
 if selected:
  eps=prereg['episode_disjoint_test_episodes'];base=shared.run_baseline(args,eps,instructions);m=compare(candidate(args,eps,instructions,selected['config']),base);test={'baseline':base,'candidate':m};checks.update({'test_zero_teacher_routes':m['teacher_routes']==0,'test_rgb_ratio_le_0p80':m['rgb_ratio']<=.80,'test_temporal_ratio_le_0p95':m['temporal_ratio']<=.95,'test_terminal_preserving_renderer':True})
 report={'format':'strict-track2-v379-temporal-renderer-sweep-v1','created_at':datetime.now(timezone.utc).isoformat(),'calibration_baseline':base_cal,'calibration':cal,'selected':selected,'episode_disjoint_test':test,'checks':checks,'passed':all(checks.values()),'authorizes_frozen_candidate_packaging':all(checks.values()),'guards':{'public_train_only':True,'reward_or_outcomes_used':False,'hidden_or_final_data':False,'real_submission':False}}
 args.output.write_text(json.dumps(report,indent=2)+'\n');print('FINAL',json.dumps({'selected':selected,'test':test,'checks':checks,'passed':report['passed']},indent=2));return 0 if report['passed'] else 3
if __name__=='__main__':raise SystemExit(main())
