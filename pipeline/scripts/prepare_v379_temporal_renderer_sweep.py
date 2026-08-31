#!/usr/bin/env python3
"""Preregister the sole remaining v378 fix: intermediate-frame smoothing."""
from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path("/root/autodl-tmp/IROS_WAM_2.0 challenge");J=ROOT/"artifacts/strict_track2_joint_augmentation_20260810";O=ROOT/"artifacts/strict_track2_official_20260810";NAME="v379_temporal_renderer_sweep_seed1542_20260823";RUN=J/NAME;REG=O/"run_registry"/NAME
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 if RUN.exists() or REG.exists():raise FileExistsError('refusing overwrite')
 rejection=J/"v378_source_routed_blended_cartesian_seed1541_20260823/audit/recursive_reward_causal.json";script=ROOT/"pipeline/scripts/audit_v379_temporal_renderer_sweep.py"
 for p in (rejection,script):
  if not p.exists():raise FileNotFoundError(p)
 x=json.loads(rejection.read_text())
 if x.get('passed') is not False or not all(x['teacher_frame_bit_exact'].values()):raise RuntimeError('unexpected v378 state')
 RUN.mkdir(parents=True);(RUN/'audit').mkdir();REG.mkdir(parents=True)
 payload={'format':'strict-track2-v379-temporal-renderer-sweep-preregistration-v1','created_at':datetime.now(timezone.utc).isoformat(),'fixed_parent':{'threshold':0.05,'cartesian_alpha':0.75,'terminal_frame':'bit-exact v378 terminal frame'},'calibration_episodes':[15,28],'episode_disjoint_test_episodes':[40,49],'configs':[{'raw_intermediate_weight':x} for x in (0.0,0.25,0.5)],'renderer':'for routed frames only, blend raw intermediate prediction with a linear ramp from request last frame to raw terminal; terminal remains exact','selection':{'reward_or_outcomes_used':False,'eligibility':'teacher routes zero, RGB ratio <=.80, temporal ratio <=.95','score':'min max(rgb_ratio/.75, temporal_ratio/.90)','tie_break':'higher raw weight'},'test_acceptance':{'RGB_ratio_le_0p80':True,'temporal_ratio_le_0p95':True,'teacher_routes_zero':True},'evidence_sha256':{'v378_rejection':sha(rejection),'script':sha(script)},'guards':{'public_train_only':True,'reward_or_outcomes_used':False,'hidden_or_final_data':False,'real_submission':False}}
 text=json.dumps(payload,indent=2)+'\n';(RUN/'preregistration.json').write_text(text);(REG/'preregistration.json').write_text(text);print(json.dumps({'run':str(RUN),'configs':payload['configs']},indent=2))
if __name__=='__main__':main()
