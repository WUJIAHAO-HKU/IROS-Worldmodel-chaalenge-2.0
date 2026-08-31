#!/usr/bin/env python3
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
BASE=Path('/root/autodl-tmp/IROS_WAM_2.0 challenge');OFF=BASE/'artifacts/strict_track2_official_20260810';JOINT=BASE/'artifacts/strict_track2_joint_augmentation_20260810';NAME='v243b_public_clean_successor_reward_map_seed1445_20260819'
def sha(p):
 h=hashlib.sha256();f=p.open('rb')
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
 f.close();return h.hexdigest()
def main():
 reg=OFF/'run_registry'/NAME;run=JOINT/NAME
 if reg.exists() or run.exists():raise SystemExit('refusing overwrite')
 reg.mkdir(parents=True);run.mkdir(parents=True);(run/'audit').mkdir()
 src=[BASE/'pipeline/scripts/map_v243_clean_successor_rewards.py',BASE/'pipeline/scripts/launch_v243_clean_reward_map.sh']
 x={'format':'strict-track2-v243-public-clean-reward-map-preregistration-v1','registered_at':datetime.now(timezone.utc).isoformat(),
 'objective':'measure reward recognition and temporal progression in official public clean demonstration successors',
 'fixed_subset':'capture_success absent and source contains aloha-agilex_clean_50; expected 1926 rows','instruction_aggregation':'mean across exactly four public v211 capture instructions','fixed_offsets':[4,8,12,16,24,32],
 'source_sha256':{str(p):sha(p) for p in src},'guards':{'public_data_only':True,'policy_modified':False,'hidden_or_final_data':False,'real_submission':False}}
 (reg/'preregistration.json').write_text(json.dumps(x,indent=2)+'\n');(run/'preregistration.json').write_text(json.dumps(x,indent=2)+'\n')
if __name__=='__main__':main()
