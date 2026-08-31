#!/usr/bin/env python3
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
B=Path('/root/autodl-tmp/IROS_WAM_2.0 challenge');O=B/'artifacts/strict_track2_official_20260810';J=B/'artifacts/strict_track2_joint_augmentation_20260810';N='v256_v254_delta_regime_long128_seed1458_20260819'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 r=O/'run_registry'/N;run=J/N
 if r.exists() or run.exists():raise SystemExit('refusing overwrite')
 capture=J/'v255_v254_delta_regime_service_seed1457_20260819/audit/post_grasp_reward_report.json'
 if json.loads(capture.read_text()).get('passed') is not True:raise ValueError('v255 capture gate did not pass')
 r.mkdir(parents=True);(run/'audit').mkdir(parents=True);(run/'local_dev_token.txt').write_text('local-dev-token\n')
 src=[B/'pipeline/wam_pipeline/v254_delta_regime_terminal_runtime.py',B/'pipeline/wam_pipeline/backends.py',B/'pipeline/scripts/export_v217_service_recursive_cache.py',B/'pipeline/scripts/summarize_v256_long_gate.py',B/'pipeline/scripts/restart_v256_services.sh',B/'pipeline/scripts/launch_v256_long_gate.sh']
 x={'format':'strict-track2-v256-long128-preregistration-v1','registered_at':datetime.now(timezone.utc).isoformat(),'frozen_model_version':'track2-v254-delta-regime-terminal','input_capture_gate':str(capture),'fixed_audit':{'chunks':16,'threshold':.9,'success_hit_rate_min':.75,'failure_hit_rate_max':.2,'margin_min':.55,'official_http_acceptance':True},'implementation':{str(p):sha(p) for p in src},'guards':{'public_data_only':True,'policy_training':False,'hidden_or_final_data':False,'real_submission':False}}
 text=json.dumps(x,indent=2)+'\n';(r/'preregistration.json').write_text(text);(run/'preregistration.json').write_text(text)
if __name__=='__main__':main()
