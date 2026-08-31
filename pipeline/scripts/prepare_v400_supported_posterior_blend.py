#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path('/root/autodl-tmp/IROS_WAM_2.0 challenge');J=ROOT/'artifacts/strict_track2_joint_augmentation_20260810';O=ROOT/'artifacts/strict_track2_official_20260810';NAME='v400_supported_posterior_blend_seed1560_20260823';RUN=J/NAME;REG=O/'run_registry'/NAME
def sha(p:Path)->str:
 d=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):d.update(b)
 return d.hexdigest()
def main()->None:
 if RUN.exists() or REG.exists():raise FileExistsError('refusing overwrite')
 base=J/'v390_continuous_phase_clean_reanchor_seed1551_20260823/release';phase=base/'recursive_phase_gate.npz';sources={'v399_audit':J/'v399_posterior_blend_clean_reanchor_seed1559_20260823/audit/recursive_reward_causal.json','v399_runtime':ROOT/'pipeline/wam_pipeline/v399_posterior_blend_clean_reanchor_runtime.py','runtime':ROOT/'pipeline/wam_pipeline/v400_supported_posterior_blend_runtime.py','backend':ROOT/'pipeline/wam_pipeline/backends.py','v389_training':J/'v389r1_public_recursive_phase_classifier_seed1550_20260823/training_report.json','phase_gate':phase}
 missing=[str(p) for p in sources.values() if not p.is_file()]
 if missing:raise FileNotFoundError(missing)
 audit=json.loads(sources['v399_audit'].read_text());training=json.loads(sources['v389_training'].read_text())
 if audit.get('passed') is not False or audit['success_classification']['validation']['candidate']['false_positive']!=21:raise RuntimeError('v399 rejection evidence drift')
 if .5 not in [float(r['threshold']) for r in training['cv_rows']]:raise RuntimeError('0.5 was not a declared classifier threshold')
 RUN.mkdir(parents=True);(RUN/'audit').mkdir();REG.mkdir(parents=True);release=RUN/'release';release.mkdir()
 for name in ('left_expert','right_expert','arm_routed_autoregressive_manifest.json','source_gate.npz','native_batch_clean_reanchor_manifest.json','recursive_phase_gate.npz','continuous_phase_clean_reanchor_manifest.json'):
  source=base/name;os.symlink(source.resolve(),release/name,target_is_directory=source.is_dir())
 manifest={'format':'strict-track2-v400-supported-posterior-blend-release-v1','model_version':'track2-v400-v390-supported-posterior-blend','phase_gate':'recursive_phase_gate.npz','phase_gate_sha256':sha(phase),'support_probability_min':.5,'blend_formula':'alpha = posterior if posterior >= 0.5 else 0','support_rationale':'0.5 is the minimum predeclared v389 classifier operating point; v399 sole added public-validation false positive was outside support at p=0.4267','no_parameter_sweep':True,'reward_or_outcomes_used':False,'hidden_or_final_data':False};(release/'supported_posterior_blend_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 prereg={'format':'strict-track2-v400-supported-posterior-blend-preregistration-v1','registered_at':datetime.now(timezone.utc).isoformat(),'model_version':manifest['model_version'],'release':str(release),'fixed_candidate':manifest,'fixed_recursive_gate':{'exact_rows':512,'teacher_bit_exact_v326':True,'recursive_rgb_ratio_max_both':1.05,'recursive_temporal_ratio_max_both':1.05,'recursive_reward_error_ratio_lt_both':1.0,'positive_recall_min_both':.50,'positive_recall_nonregression_vs_v326':True,'false_positive_rate_nonregression_vs_v326':True,'all_required':True},'next_authority':'passing audit authorizes one fresh 32 zero-update preflight only','evidence_sha256':{k:sha(p) for k,p in sources.items()},'guards':{'public_world_model_data_only':True,'runtime_reads_reward_or_outcomes':False,'policy_modified':False,'hidden_or_final_data':False,'real_submission':False}}
 text=json.dumps(prereg,indent=2)+'\n';(RUN/'release_registration.json').write_text(text);(REG/'preregistration.json').write_text(text);print(json.dumps({'run':str(RUN),'release':str(release)},indent=2))
if __name__=='__main__':main()
