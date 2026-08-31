#!/usr/bin/env python3
"""Preregister all public gates for the independently frozen v301 runtime."""
from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
B=Path('/root/autodl-tmp/IROS_WAM_2.0 challenge'); O=B/'artifacts/strict_track2_official_20260810'; J=B/'artifacts/strict_track2_joint_augmentation_20260810'; N='v303_v301_batched_gates_seed1482_20260821'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''): h.update(block)
 return h.hexdigest()
def main():
 reg,run=O/'run_registry'/N,J/N
 if reg.exists() or run.exists(): raise FileExistsError('refusing overwrite')
 batch_path=O/'diagnostics/v302_v301_batch_equivalence_20260821.json'; batch=json.loads(batch_path.read_text())
 if batch['max_absolute_pixel_change']>2 or batch['mean_absolute_pixel_change']>.05 or batch['speedup']<1.5: raise RuntimeError('batch numerical/throughput gate failed')
 pixel_path=O/'diagnostics/v296_v295_terminal_frame_equivariance_20260821.json'; pixel=json.loads(pixel_path.read_text())
 sources=[B/'pipeline/wam_pipeline/v301_batched_terminal_frame_mirror_runtime.py',B/'pipeline/wam_pipeline/v295_terminal_frame_preserving_mirror_runtime.py',B/'pipeline/wam_pipeline/backends.py',B/'pipeline/scripts/restart_v301_services.sh',B/'pipeline/scripts/launch_v303_v301_gates.sh',B/'pipeline/scripts/verify_v298_capture_relative.py']
 reg.mkdir(parents=True); (run/'audit').mkdir(parents=True); (run/'local_dev_token.txt').write_text('local-dev-token\n')
 payload={'format':'strict-track2-v303-v301-batched-gates-preregistration-v1','registered_at':datetime.now(timezone.utc).isoformat(),'fixed_model':{'model_version':'track2-v301-batched-terminal-frame-mirror-v295','logic':'v295 terminal-frame preserving mirror','implementation':'native parent batching','response':'RGB only'},'batch_equivalence_gate':{'max_absolute_pixel_change':2,'mean_absolute_pixel_change':.05,'speedup_min':1.5,'observed':batch},'v295_ground_truth_evidence':{'path':str(pixel_path),'sha256':sha(pixel_path),'validation_relative_mae_change':pixel['summaries']['validation']['relative_mae_change'],'local_relative_mae_change':pixel['summaries']['local_test']['relative_mae_change'],'triangle_bound_extra_mae':batch['mean_absolute_pixel_change']},'fixed_audit':{'recursive_chunks':16,'threshold':.9,'success_hit_rate_min':.75,'failure_hit_rate_max':.20,'margin_min':.55},'input_capture_gate':str(run/'audit/capture_relative_gate.json'),'implementation':{str(p):sha(p) for p in sources},'guards':{'participant_component':'world-model RGB service only','policy_modified':False,'runtime_uses_reward':False,'public_data_only':True,'hidden_or_final_data':False,'real_submission':False}}
 text=json.dumps(payload,indent=2)+'\n'; (reg/'preregistration.json').write_text(text); (run/'release_registration.json').write_text(text)
if __name__=='__main__': main()
