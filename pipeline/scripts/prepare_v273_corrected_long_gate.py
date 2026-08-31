#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
B=Path('/root/autodl-tmp/IROS_WAM_2.0 challenge');O=B/'artifacts/strict_track2_official_20260810';J=B/'artifacts/strict_track2_joint_augmentation_20260810';N='v273_v271_corrected_alpha_long_gate_seed1470_20260819'
def sha(p):
 d=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):d.update(b)
 return d.hexdigest()
def main():
 reg=O/'run_registry'/N;run=J/N
 if reg.exists() or run.exists():raise FileExistsError('refusing overwrite')
 source=J/'v272_v271_endpoint_calibrated_gates_seed1469_20260819/audit/post_grasp_reward_report.json';x=json.loads(source.read_text())
 assert x['passed'] is False and [k for k,v in x['checks'].items() if not v]==['regime_alpha_global']
 sources=[B/'pipeline/scripts/audit_v273_corrected_endpoint_alpha.py',B/'pipeline/wam_pipeline/v271_endpoint_calibrated_terminal_runtime.py',B/'pipeline/wam_pipeline/backends.py',B/'pipeline/scripts/restart_v271_v273_services.sh',B/'pipeline/scripts/launch_v273_corrected_long_gate.sh',B/'pipeline/scripts/summarize_v256_long_gate.py']
 reg.mkdir(parents=True);(run/'audit').mkdir(parents=True);(run/'local_dev_token.txt').write_text('local-dev-token\n')
 capture=run/'audit/corrected_endpoint_alpha_report.json'
 payload={'format':'strict-track2-v273-corrected-alpha-long-gate-preregistration-v1','registered_at':datetime.now(timezone.utc).isoformat(),'frozen_model_version':'track2-v271-endpoint-calibrated-terminal','source_stale_alpha_report':str(source),'source_stale_alpha_sha256':sha(source),'corrected_capture_gate':{'output':str(capture),'actual_alpha_reward_correlation_min':.70,'ood_endpoint_correlation_min':.50,'ood_progress_correlation_min':.50,'success_like_min':4},'input_capture_gate':str(capture),'fixed_audit':{'chunks':16,'threshold':.9,'success_hit_rate_min':.75,'failure_hit_rate_max':.20,'margin_min':.55,'official_http_acceptance':True},'implementation':{str(p):sha(p) for p in sources},'guards':{'public_data_only':True,'policy_training':False,'hidden_or_final_data':False,'real_submission':False}}
 text=json.dumps(payload,indent=2)+'\n';(reg/'preregistration.json').write_text(text);(run/'release_registration.json').write_text(text)
if __name__=='__main__':main()

