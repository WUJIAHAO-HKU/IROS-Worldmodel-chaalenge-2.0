#!/usr/bin/env python3
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
B=Path('/root/autodl-tmp/IROS_WAM_2.0 challenge');O=B/'artifacts/strict_track2_official_20260810';J=B/'artifacts/strict_track2_joint_augmentation_20260810';N='v251_v250_specific_terminal_service_seed1454_20260819'
def sha(p):
 h=hashlib.sha256();f=p.open('rb')
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
 f.close();return h.hexdigest()
def main():
 r=O/'run_registry'/N;run=J/N
 if r.exists() or run.exists():raise SystemExit('refusing overwrite')
 r.mkdir(parents=True);run.mkdir(parents=True);(run/'audit').mkdir();(run/'local_dev_token.txt').write_text('local-dev-token\n');src=[B/'pipeline/wam_pipeline/v250_specific_terminal_successor_runtime.py',B/'pipeline/wam_pipeline/v247_terminal_successor_runtime.py',B/'pipeline/wam_pipeline/backends.py',B/'pipeline/scripts/replay_v245_post_grasp_reward.py',B/'pipeline/scripts/restart_v250_services.sh',B/'pipeline/scripts/launch_v251_service_audit.sh'];x={'format':'strict-track2-v251-service-preregistration-v1','registered_at':datetime.now(timezone.utc).isoformat(),'fixed_model':{'model_version':'track2-v250-specific-terminal-successor','alignment_floor':.5,'distance_temperature':.5,'alpha_scale':16.,'target_rule':'visual public-clean terminal successor'},'fixed_gate':{'success_like_min':8,'group_std_min':.01,'global_min':.2,'group_min':.45,'service_contract':True},'inputs':{'v249b_proxy':str(O/'run_registry/v249b_terminal_alignment_floor_proxy_seed1453_20260819/analysis_report.json'),'v248_false_positive_analysis':str(O/'run_registry/v248_v247_terminal_successor_long128_seed1451_20260819/false_positive_gate_report.json')},'implementation':{str(p):sha(p) for p in src},'guards':{'runtime_uses_reward':False,'public_data_only':True,'policy_modified':False,'hidden_or_final_data':False,'real_submission':False}};(r/'preregistration.json').write_text(json.dumps(x,indent=2)+'\n');(run/'release_registration.json').write_text(json.dumps(x,indent=2)+'\n')
if __name__=='__main__':main()
