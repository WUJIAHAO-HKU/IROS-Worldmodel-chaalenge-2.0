#!/usr/bin/env python3
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
B=Path('/root/autodl-tmp/IROS_WAM_2.0 challenge');O=B/'artifacts/strict_track2_official_20260810';J=B/'artifacts/strict_track2_joint_augmentation_20260810';N='v247_v246_terminal_successor_service_seed1450_20260819'
def sha(p):
 h=hashlib.sha256();f=p.open('rb')
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
 f.close();return h.hexdigest()
def main():
 r=O/'run_registry'/N;run=J/N
 if r.exists() or run.exists():raise SystemExit('refusing overwrite')
 r.mkdir(parents=True);run.mkdir(parents=True);(run/'audit').mkdir();(run/'local_dev_token.txt').write_text('local-dev-token\n')
 src=[B/'pipeline/wam_pipeline/v247_terminal_successor_runtime.py',B/'pipeline/wam_pipeline/v245_clean_progressive_successor_runtime.py',B/'pipeline/wam_pipeline/backends.py',B/'pipeline/scripts/replay_v245_post_grasp_reward.py',B/'pipeline/scripts/restart_v247_services.sh',B/'pipeline/scripts/launch_v247_service_audit.sh']
 x={'format':'strict-track2-v247-service-preregistration-v1','registered_at':datetime.now(timezone.utc).isoformat(),'fixed_model':{'model_version':'track2-v247-public-clean-terminal-successor','target_rule':'visually nearest official public clean episode, final available row at start>=112','action_weight':4.0,'distance_scale':2.9357216358184814,'motion_scale':.16446852684020996,'alpha_scale':4.0,'alignment_gate':'smooth'},'fixed_gate':{'success_like_min':8,'group_std_min':.01,'global_expert_alignment_min':.15,'motion_global_min':.05,'group_concordance_min':.55,'http_acceptance':True},'input_proxy':str(O/'run_registry/v246_public_clean_terminal_proxy_seed1449_20260819/analysis_report.json'),'implementation':{str(p):sha(p) for p in src},'guards':{'runtime_uses_reward':False,'official_public_clean_demonstrations_only':True,'policy_modified':False,'hidden_or_final_data':False,'real_submission':False}};(r/'preregistration.json').write_text(json.dumps(x,indent=2)+'\n');(run/'release_registration.json').write_text(json.dumps(x,indent=2)+'\n')
if __name__=='__main__':main()
