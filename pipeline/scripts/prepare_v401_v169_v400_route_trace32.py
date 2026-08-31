#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path('/root/autodl-tmp/IROS_WAM_2.0 challenge');O=ROOT/'artifacts/strict_track2_official_20260810';J=ROOT/'artifacts/strict_track2_joint_augmentation_20260810';NAME='v401_v169_v400_route_trace32_seed1561_20260823';REG=O/'run_registry'/NAME;RUN=O/'runs'/NAME
def sha(p:Path)->str:
 d=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):d.update(b)
 return d.hexdigest()
def main()->None:
 if REG.exists() or RUN.exists():raise FileExistsError('refusing overwrite')
 paths={'v400_fidelity':J/'v400_supported_posterior_blend_seed1560_20260823/audit/recursive_reward_causal.json','v400_manifest':J/'v400_supported_posterior_blend_seed1560_20260823/release/supported_posterior_blend_manifest.json','runtime':ROOT/'pipeline/wam_pipeline/v401_v400_route_trace_runtime.py','backend':ROOT/'pipeline/wam_pipeline/backends.py','service':ROOT/'pipeline/scripts/restart_v401_trace_services.sh','launcher':ROOT/'pipeline/scripts/launch_v401_v169_v400_route_trace32.sh','analyzer':ROOT/'pipeline/scripts/analyze_v401_route_trace.py','frozen_v169':O/'runs/v169_parent_step5_seed1243_20260813/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt'}
 missing=[str(p) for p in paths.values() if not p.is_file()]
 if missing:raise FileNotFoundError(missing)
 if not json.loads(paths['v400_fidelity'].read_text()).get('passed'):raise RuntimeError('v400 fidelity gate failed')
 REG.mkdir(parents=True);payload={'format':'strict-track2-v401-v169-v400-route-trace32-preregistration-v1','registered_at':datetime.now(timezone.utc).isoformat(),'run_path':str(RUN),'purpose':'one fresh zero-update preflight for frozen v400 before any RL','protocol':{'trajectories':32,'rollout_epochs':1,'episode_steps':200,'actor_seed':1561,'environment_seed':0,'policy_updates':0,'checkpoint_writes':0,'world_model_outputs':'bit-equivalent v400; telemetry side effect only'},'gate':{'minimum_successes':13,'rationale':'nearest integer non-regression threshold to historical frozen 51/128','minimum_supported_routes':4,'all_required':True},'on_pass':'authorize exactly one conservative RL update; no public batch16 or final evaluation yet','on_fail':'reject v400 before RL; stop continuous-phase branch','evidence_sha256':{k:sha(p) for k,p in paths.items()},'guards':{'public_world_model_rollout_only':True,'policy_modified':False,'reward_modified':False,'hidden_or_final_data':False,'real_submission':False}};(REG/'preregistration.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps({'registry':str(REG),'run':str(RUN)},indent=2))
if __name__=='__main__':main()
