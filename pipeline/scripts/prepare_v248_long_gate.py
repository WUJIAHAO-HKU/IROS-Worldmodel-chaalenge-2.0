#!/usr/bin/env python3
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
B=Path('/root/autodl-tmp/IROS_WAM_2.0 challenge');O=B/'artifacts/strict_track2_official_20260810';J=B/'artifacts/strict_track2_joint_augmentation_20260810';N='v248_v247_terminal_successor_long128_seed1451_20260819'
def sha(p):
 h=hashlib.sha256();f=p.open('rb')
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
 f.close();return h.hexdigest()
def main():
 r=O/'run_registry'/N;run=J/N
 if r.exists() or run.exists():raise SystemExit('refusing overwrite')
 r.mkdir(parents=True);run.mkdir(parents=True);(run/'audit').mkdir();(run/'local_dev_token.txt').write_text('local-dev-token\n')
 src=[B/'pipeline/wam_pipeline/v247_terminal_successor_runtime.py',B/'pipeline/scripts/export_v217_service_recursive_cache.py',B/'pipeline/scripts/summarize_v248_long_gate.py',B/'pipeline/scripts/restart_v247_services.sh',B/'pipeline/scripts/launch_v248_long_gate.sh']
 x={'format':'strict-track2-v248-long128-preregistration-v1','registered_at':datetime.now(timezone.utc).isoformat(),'frozen_model_version':'track2-v247-public-clean-terminal-successor','input_capture_gate':str(J/'v247_v246_terminal_successor_service_seed1450_20260819/audit/post_grasp_reward_report.json'),'fixed_audit':{'chunks':16,'threshold':.9,'success_hit_rate_min':.75,'failure_hit_rate_max':.2,'margin_min':.55,'official_http_acceptance':True},'implementation':{str(p):sha(p) for p in src},'guards':{'public_data_only':True,'policy_training':False,'hidden_or_final_data':False,'real_submission':False}};(r/'preregistration.json').write_text(json.dumps(x,indent=2)+'\n');(run/'preregistration.json').write_text(json.dumps(x,indent=2)+'\n')
if __name__=='__main__':main()
