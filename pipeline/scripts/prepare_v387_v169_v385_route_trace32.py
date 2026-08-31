#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path('/root/autodl-tmp/IROS_WAM_2.0 challenge'); O=ROOT/'artifacts/strict_track2_official_20260810'; J=ROOT/'artifacts/strict_track2_joint_augmentation_20260810'; NAME='v387_v169_v385_route_trace32_seed1497_20260823'; REG=O/'run_registry'/NAME; RUN=O/'runs'/NAME
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 if REG.exists() or RUN.exists():raise FileExistsError('refusing overwrite')
 paths={'v385_gate':J/'v385_native_batch_clean_reanchor_seed1548_20260823/audit/recursive_reward_causal.json','v386_result':O/'runs/v386_v169_v385_trainmode_rolloutonly128_seed1497_20260823/audit/rollout_go_no_go.json','runtime':ROOT/'pipeline/wam_pipeline/v387_v385_route_trace_runtime.py','backend':ROOT/'pipeline/wam_pipeline/backends.py','launcher':ROOT/'pipeline/scripts/launch_v387_v169_v385_route_trace32.sh'}
 for p in paths.values():
  if not p.is_file():raise FileNotFoundError(p)
 REG.mkdir(parents=True)
 payload={'format':'strict-track2-v387-v169-v385-route-trace32-preregistration-v1','registered_at':datetime.now(timezone.utc).isoformat(),'run_path':str(RUN),'purpose':'diagnose which fixed v385 route stage rejects frozen v169 policy actions','protocol':{'trajectories':32,'rollout_epochs':1,'episode_steps':200,'actor_seed':1497,'environment_seed':0,'policy_updates':0,'checkpoint_writes':0,'world_model_outputs':'bit-equivalent v385; telemetry side effect only'},'telemetry':['right','source_ready','post_grasp','action_probability_ge_0p99','failure_signature','action_phase_ready','final_route'],'selection_authority':False,'evidence_sha256':{k:sha(p) for k,p in paths.items()},'guards':{'public_world_model_rollout_only':True,'policy_modified':False,'reward_modified':False,'hidden_or_final_data':False,'real_submission':False}}
 (REG/'preregistration.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps({'registry':str(REG),'run':str(RUN)},indent=2))
if __name__=='__main__':main()
