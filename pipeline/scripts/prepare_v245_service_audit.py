#!/usr/bin/env python3
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
BASE=Path('/root/autodl-tmp/IROS_WAM_2.0 challenge');OFF=BASE/'artifacts/strict_track2_official_20260810';JOINT=BASE/'artifacts/strict_track2_joint_augmentation_20260810';NAME='v245b_v244_clean_progressive_service_seed1448_20260819'
def sha(p):
 h=hashlib.sha256();f=p.open('rb')
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
 f.close();return h.hexdigest()
def main():
 reg=OFF/'run_registry'/NAME;run=JOINT/NAME
 if reg.exists() or run.exists():raise SystemExit('refusing overwrite')
 reg.mkdir(parents=True);run.mkdir(parents=True);(run/'audit').mkdir();(run/'local_dev_token.txt').write_text('local-dev-token\n')
 src=[BASE/'pipeline/wam_pipeline/v245_clean_progressive_successor_runtime.py',BASE/'pipeline/wam_pipeline/backends.py',BASE/'pipeline/scripts/replay_v245_post_grasp_reward.py',BASE/'pipeline/scripts/restart_v245_services.sh',BASE/'pipeline/scripts/launch_v245_service_audit.sh']
 x={'format':'strict-track2-v245-service-preregistration-v1','registered_at':datetime.now(timezone.utc).isoformat(),'fixed_model':{'model_version':'track2-v245b-public-clean-progressive-offset24','action_weight':4.0,'progress_offset':24,'distance_scale':2.9357216358184814,'motion_scale':.16446852684020996,'alpha_scale':2.0,'pregrasp_alpha':.70,'clean_partition':'public episode_id >= 20000'},'fixed_gate':{'group_std_min':1e-4,'success_like_count_min':4,'expert_global_min':.05,'alignment_global_min':.05,'expert_group_concordance_min':.55,'alignment_group_concordance_min':.55,'official_http_acceptance':True},'inputs':{'v244_proxy':str(OFF/'run_registry/v244_public_clean_progressive_proxy_seed1446_20260819/analysis_diagnostics3.json'),'v243_map':str(JOINT/'v243b_public_clean_successor_reward_map_seed1445_20260819/audit/reward_map_report.json')},'implementation':{str(p):sha(p) for p in src},'guards':{'runtime_uses_reward':False,'official_public_clean_demonstrations_only':True,'policy_modified':False,'hidden_or_final_data':False,'real_submission':False}}
 (reg/'preregistration.json').write_text(json.dumps(x,indent=2)+'\n');(run/'release_registration.json').write_text(json.dumps({'format':'strict-track2-v245-release-v1','model':x['fixed_model'],'implementation':x['implementation'],'guards':x['guards']},indent=2)+'\n')
if __name__=='__main__':main()
