#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path

BASE = Path('/root/autodl-tmp/IROS_WAM_2.0 challenge')
OFF = BASE / 'artifacts/strict_track2_official_20260810'
JOINT = BASE / 'artifacts/strict_track2_joint_augmentation_20260810'
NAME = 'v242b_public_reward_blend_scale_sweep_seed1443_20260819'

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def main() -> None:
    reg, run = OFF / 'run_registry' / NAME, JOINT / NAME
    if reg.exists() or run.exists(): raise SystemExit('refusing to overwrite v242b')
    reg.mkdir(parents=True); run.mkdir(parents=True); (run / 'audit').mkdir()
    sources = [BASE / 'pipeline/scripts/sweep_v242_service_blend_scale.py',
               BASE / 'pipeline/scripts/launch_v242_reward_scale_sweep.sh',
               BASE / 'pipeline/scripts/restart_v241_services.sh',
               BASE / 'pipeline/scripts/restart_v209_services.sh']
    payload = {
      'format':'strict-track2-v242-public-reward-scale-preregistration-v1',
      'registered_at':datetime.now(timezone.utc).isoformat(),
      'objective':'diagnose whether v241 target selection or blend magnitude reverses public post-grasp reward ranking',
      'fixed_scales':[0.0,0.5,1.0,2.0,4.0,8.0],
      'selection_rule':'group std >=1e-5 and reward correlations with expert similarity, motion, alignment all >=0.05; maximize minimum correlation',
      'inputs':{'capture_registry':'v211_v209_train_action_capture_h200_r2_step1_seed1410_20260818',
                'parent_release':'v209_v202_v208_public_arm_routed_release',
                'candidate_release':'v241b_v240_alignment_gated_transport_service_seed1441_20260819'},
      'source_sha256':{str(p):sha(p) for p in sources},
      'guards':{'public_data_only':True,'policy_modified':False,'hidden_or_final_data':False,'real_submission':False}}
    (reg/'preregistration.json').write_text(json.dumps(payload,indent=2)+'\n')
    (run/'preregistration.json').write_text(json.dumps(payload,indent=2)+'\n')

if __name__ == '__main__': main()
