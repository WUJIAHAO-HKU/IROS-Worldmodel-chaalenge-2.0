#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path('/root/autodl-tmp/IROS_WAM_2.0 challenge');O=ROOT/'artifacts/strict_track2_official_20260810';J=ROOT/'artifacts/strict_track2_joint_augmentation_20260810';NAME='v398_v169_v390_relative_action_trace32_seed1558_20260823';REG=O/'run_registry'/NAME;RUN=O/'runs'/NAME
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def main()->None:
 if REG.exists() or RUN.exists():raise FileExistsError('refusing overwrite')
 paths={'v390_fidelity':J/'v390_continuous_phase_clean_reanchor_seed1551_20260823/audit/recursive_reward_causal.json','v397_training':J/'v397_relative_action_terminal_rescue_seed1557_20260823/training_report.json','v397_v211_diagnosis':J/'v397_relative_action_terminal_rescue_seed1557_20260823/outcome_free_v211_diagnosis.json','runtime':ROOT/'pipeline/wam_pipeline/v398_v390_relative_action_trace_runtime.py','backend':ROOT/'pipeline/wam_pipeline/backends.py','service':ROOT/'pipeline/scripts/restart_v398_trace_services.sh','launcher':ROOT/'pipeline/scripts/launch_v398_v169_relative_action_trace32.sh','analyzer':ROOT/'pipeline/scripts/analyze_v398_relative_action_trace.py','frozen_v169':O/'runs/v169_parent_step5_seed1243_20260813/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt'}
 missing=[str(p) for p in paths.values() if not p.is_file()]
 if missing:raise FileNotFoundError(missing)
 if not json.loads(paths['v390_fidelity'].read_text()).get('passed') or not json.loads(paths['v397_training'].read_text()).get('passed'):raise RuntimeError('source gate did not pass')
 REG.mkdir(parents=True)
 payload={'format':'strict-track2-v398-v169-v390-relative-action-trace32-preregistration-v1','registered_at':datetime.now(timezone.utc).isoformat(),'run_path':str(RUN),'purpose':'disambiguate v397 zero coverage on failure-heavy v211 by measuring its outcome-free probability on the exact frozen v169 request distribution while preserving v390 outputs','protocol':{'trajectories':32,'rollout_epochs':1,'episode_steps':200,'actor_seed':1558,'environment_seed':0,'policy_updates':0,'checkpoint_writes':0,'world_model_outputs':'bit-equivalent v390; relative-action telemetry side effect only','raw_success_metrics':'sequestered and forbidden for model selection'},'gate':{'minimum_eligible_requests':20,'minimum_relative_routes':4,'union_must_exceed_old':True,'all_required':True},'on_pass':'authorize one v399 OR-rescue integration and 512 public fidelity audit only','on_fail':'reject relative-action rescue; do not tune its threshold from this trace','evidence_sha256':{k:sha(p) for k,p in paths.items()},'guards':{'public_world_model_rollout_only':True,'outcomes_used_for_selection':False,'policy_modified':False,'reward_modified':False,'hidden_or_final_data':False,'real_submission':False}}
 (REG/'preregistration.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps({'registry':str(REG),'run':str(RUN)},indent=2))
if __name__=='__main__':main()
