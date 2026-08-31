#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path('/root/autodl-tmp/IROS_WAM_2.0 challenge');J=ROOT/'artifacts/strict_track2_joint_augmentation_20260810';O=ROOT/'artifacts/strict_track2_official_20260810';NAME='v399_posterior_blend_clean_reanchor_seed1559_20260823';RUN=J/NAME;REG=O/'run_registry'/NAME
def sha(p:Path)->str:
 d=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):d.update(block)
 return d.hexdigest()
def main()->None:
 if RUN.exists() or REG.exists():raise FileExistsError('refusing overwrite')
 base=J/'v390_continuous_phase_clean_reanchor_seed1551_20260823/release';phase=base/'recursive_phase_gate.npz';runtime=ROOT/'pipeline/wam_pipeline/v399_posterior_blend_clean_reanchor_runtime.py';backend=ROOT/'pipeline/wam_pipeline/backends.py';trace=O/'run_registry/v398_v169_v390_relative_action_trace32_seed1558_20260823/relative_action_analysis.json'
 sources={'base_manifest':base/'continuous_phase_clean_reanchor_manifest.json','phase_gate':phase,'runtime':runtime,'backend':backend,'v390_fidelity':J/'v390_continuous_phase_clean_reanchor_seed1551_20260823/audit/recursive_reward_causal.json','v398_outcome_free_trace':trace}
 missing=[str(p) for p in sources.values() if not p.is_file()]
 if missing:raise FileNotFoundError(missing)
 if not json.loads(sources['v390_fidelity'].read_text()).get('passed'):raise RuntimeError('v390 fidelity drift')
 RUN.mkdir(parents=True);(RUN/'audit').mkdir();REG.mkdir(parents=True);release=RUN/'release';release.mkdir()
 for name in ('left_expert','right_expert','arm_routed_autoregressive_manifest.json','source_gate.npz','native_batch_clean_reanchor_manifest.json','recursive_phase_gate.npz','continuous_phase_clean_reanchor_manifest.json'):
  source=base/name;os.symlink(source.resolve(),release/name,target_is_directory=source.is_dir())
 manifest={'format':'strict-track2-v399-posterior-blend-clean-reanchor-release-v1','model_version':'track2-v399-v390-posterior-weighted-clean-reanchor','phase_gate':'recursive_phase_gate.npz','phase_gate_sha256':sha(phase),'blend_formula':'alpha = frozen phase posterior probability','change':'replace v390 thresholded all-or-none clean terminal replacement with posterior-weighted RGB blending for already eligible requests','unchanged_routes':'right/source/post-grasp/action>=0.99/no-failure remain v390-exact','no_parameter_sweep':True,'reward_or_outcomes_used':False,'hidden_or_final_data':False};(release/'posterior_blend_clean_reanchor_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 prereg={'format':'strict-track2-v399-posterior-blend-clean-reanchor-preregistration-v1','registered_at':datetime.now(timezone.utc).isoformat(),'model_version':manifest['model_version'],'release':str(release),'fixed_candidate':manifest,'rationale':'v398 outcome-free v169 trace shows 58 eligible requests, old posterior mass 14.32 versus 12 hard routes; posterior expectation removes discontinuity without threshold tuning','fixed_recursive_gate':{'exact_rows':512,'teacher_bit_exact_v326':True,'recursive_rgb_ratio_max_both':1.05,'recursive_temporal_ratio_max_both':1.05,'recursive_reward_error_ratio_lt_both':1.0,'positive_recall_min_both':.50,'positive_recall_nonregression_vs_v326':True,'false_positive_rate_nonregression_vs_v326':True,'all_required':True},'next_authority':'only passing 512 audit authorizes one fresh 32-trajectory zero-update preflight; never direct RL','evidence_sha256':{k:sha(p) for k,p in sources.items()},'guards':{'public_world_model_data_only':True,'v398_trace_used_without_success_or_reward':True,'runtime_reads_reward_or_outcomes':False,'policy_modified':False,'hidden_or_final_data':False,'real_submission':False}}
 text=json.dumps(prereg,indent=2)+'\n';(RUN/'release_registration.json').write_text(text);(REG/'preregistration.json').write_text(text);print(json.dumps({'run':str(RUN),'release':str(release)},indent=2))
if __name__=='__main__':main()
