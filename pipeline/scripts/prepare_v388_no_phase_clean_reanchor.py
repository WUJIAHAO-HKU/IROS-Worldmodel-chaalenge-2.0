#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path('/root/autodl-tmp/IROS_WAM_2.0 challenge');J=ROOT/'artifacts/strict_track2_joint_augmentation_20260810';O=ROOT/'artifacts/strict_track2_official_20260810';NAME='v388_no_phase_clean_reanchor_seed1549_20260823';RUN=J/NAME;REG=O/'run_registry'/NAME
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 if RUN.exists() or REG.exists():raise FileExistsError('refusing overwrite')
 base=J/'v385_native_batch_clean_reanchor_seed1548_20260823/release';trace=O/'runs/v387_v169_v385_route_trace32_seed1497_20260823/audit/route_analysis.json';runtime=ROOT/'pipeline/wam_pipeline/v388_no_phase_clean_reanchor_runtime.py';backends=ROOT/'pipeline/wam_pipeline/backends.py'
 for p in (base,trace,runtime,backends):
  if not p.exists():raise FileNotFoundError(p)
 diagnosis=json.loads(trace.read_text())
 if diagnosis['requests']!=800 or diagnosis['sequential_survivors']['phase_ready']!=1:raise RuntimeError('v387 diagnosis mismatch')
 RUN.mkdir(parents=True);(RUN/'audit').mkdir();REG.mkdir(parents=True);release=RUN/'release';release.mkdir()
 for n in ('left_expert','right_expert','arm_routed_autoregressive_manifest.json','source_gate.npz','native_batch_clean_reanchor_manifest.json'):
  p=base/n;os.symlink(p.resolve(),release/n,target_is_directory=p.is_dir())
 manifest={'format':'strict-track2-v388-no-phase-clean-reanchor-release-v1','model_version':'track2-v388-v385-no-phase-clean-reanchor','change':'remove only action-only phase-ready condition diagnosed at 1/800; retain right/source/post-grasp/action>=0.99/failure gates','reward_or_outcomes_used':False,'hidden_or_final_data':False};(release/'no_phase_clean_reanchor_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 pre={'format':'strict-track2-v388-no-phase-clean-reanchor-preregistration-v1','registered_at':datetime.now(timezone.utc).isoformat(),'model_version':manifest['model_version'],'release':str(release),'fixed_candidate':manifest['change'],'no_parameter_sweep':True,'fixed_recursive_gate':{'exact_rows':512,'teacher_bit_exact_v326':True,'recursive_rgb_ratio_max_both':1.05,'recursive_temporal_ratio_max_both':1.05,'recursive_reward_error_ratio_lt_both':1.0,'positive_recall_min_both':0.50,'positive_recall_nonregression_vs_v326':True,'false_positive_rate_nonregression_vs_v326':True,'all_required':True},'next_authority':'only 32-trajectory zero-update preflight; never direct RL','evidence_sha256':{'v387_diagnosis':sha(trace),'runtime':sha(runtime),'backends':sha(backends)},'guards':{'public_data_only':True,'runtime_reads_reward_or_outcomes':False,'policy_modified':False,'official_reward_modified':False,'hidden_or_final_data':False,'real_submission':False}}
 text=json.dumps(pre,indent=2)+'\n';(RUN/'release_registration.json').write_text(text);(REG/'preregistration.json').write_text(text);print(json.dumps({'run':str(RUN),'release':str(release)},indent=2))
if __name__=='__main__':main()
