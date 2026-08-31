#!/usr/bin/env python3
"""Preregister a public-train real-vs-generated context source detector."""
from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path('/root/autodl-tmp/IROS_WAM_2.0 challenge');J=ROOT/'artifacts/strict_track2_joint_augmentation_20260810';O=ROOT/'artifacts/strict_track2_official_20260810';NAME='v381_public_recursive_source_gate_seed1544_20260823';RUN=J/NAME;REG=O/'run_registry'/NAME
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 if RUN.exists() or REG.exists():raise FileExistsError('refusing overwrite')
 sources={'trainer':ROOT/'pipeline/scripts/train_v381_public_recursive_source_gate.py','feature_contract':ROOT/'pipeline/wam_pipeline/v337_public_recursive_ood_gate.py','v355_manifest':J/'v355_v202_v354_parametric_arm_routed_release/arm_routed_autoregressive_manifest.json','v378_manifest':J/'v378_source_routed_blended_cartesian_seed1541_20260823/release/source_routed_blend_manifest.json','split':J/'v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json'}
 for p in sources.values():
  if not p.exists():raise FileNotFoundError(p)
 RUN.mkdir(parents=True);REG.mkdir(parents=True)
 payload={'format':'strict-track2-v381-public-recursive-source-gate-preregistration-v1','created_at':datetime.now(timezone.utc).isoformat(),'purpose':'detect any model-generated recursive context rather than only severe RGB corruption','training_rule':{'episodes':'exact 15 declared public-train right episodes','alignments':list(range(8)),'negative':'all real teacher contexts and initial recursive contexts before a model prediction','positive':'all post-first-step v355 and v378 recursively generated contexts','features':'same scene-agnostic 156D RGB/gradient/temporal summaries','labels_use_reward_or_outcome':False},'selection_rule':{'grouping':'leave-one-public-train-episode-out','c_values':[0.01,0.1,1.0],'thresholds':[0.5,0.7,0.8,0.9,0.95,0.975,0.99],'eligibility':'teacher specificity ==1.0, overall generated recall >=.95, each-source recall >=.90','tie_break':'overall recall, minimum source recall, higher threshold, smaller C'},'authorizes_rl':False,'source_sha256':{k:sha(v) for k,v in sources.items()},'guards':{'public_train_only':True,'reward_or_success_outcomes_used':False,'public_holdout_read':False,'hidden_or_final_data':False,'real_submission':False}}
 text=json.dumps(payload,indent=2)+'\n';(RUN/'preregistration.json').write_text(text);(REG/'preregistration.json').write_text(text);print(json.dumps({'run':str(RUN)},indent=2))
if __name__=='__main__':main()
