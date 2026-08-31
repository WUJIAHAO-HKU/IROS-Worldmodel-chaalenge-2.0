#!/usr/bin/env python3
"""Preregister the resource-only retry of the v274 full-budget v271 RL run."""
from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
B=Path('/root/autodl-tmp/IROS_WAM_2.0 challenge');O=B/'artifacts/strict_track2_official_20260810';J=B/'artifacts/strict_track2_joint_augmentation_20260810';N='v274_v271_fresh_fullbudget_h200_r8_step5_lr2e5_beta001_seed1471_retry1_20260819';REG=O/'run_registry'/N;RUN=O/'runs'/N;GATE=J/'v273_v271_corrected_alpha_long_gate_seed1470_20260819';MANIFEST=GATE/'release_registration.json';FAILED=O/'runs/v274_v271_fresh_fullbudget_h200_r8_step5_lr2e5_beta001_seed1471_20260819';FAILED_LOG=FAILED/'launcher.log'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 if REG.exists() or RUN.exists():raise FileExistsError('refusing overwrite')
 capture=GATE/'audit/corrected_endpoint_alpha_report.json';long=GATE/'audit/long_gate_report.json'
 assert json.loads(capture.read_text())['passed'] is True and (GATE/'V273_CORRECTED_CAPTURE_GATE_PASSED').exists()
 assert json.loads(long.read_text())['passed'] is True and (GATE/'V273_LONG_GATE_PASSED').exists()
 assert 'torch.OutOfMemoryError: CUDA out of memory' in FAILED_LOG.read_text(errors='replace')
 assert not list(FAILED.glob('**/checkpoints/**/*.pt'))
 official=B/'artifacts/official_resources/pi05_adjust_bottle/model.safetensors';runner=B/'pipeline/scripts/run_strict_track2_conservative_kl.sh';runtime=B/'pipeline/wam_pipeline/v271_endpoint_calibrated_terminal_runtime.py';backends=B/'pipeline/wam_pipeline/backends.py';REG.mkdir(parents=True)
 payload={'format':'strict-track2-v274-v271-fresh-fullbudget-rl-preregistration-v2-resource-retry','timestamp_utc':datetime.now(timezone.utc).isoformat(),'purpose':'resource-only retry of fresh fixed official Pi0.5 policy training after the original run OOMed before completing its first rollout','run_path':str(RUN),'failed_attempt':{'run_path':str(FAILED),'launcher_log':str(FAILED_LOG),'launcher_log_sha256':sha(FAILED_LOG),'failure':'actor and rollout copies overlapped on GPU; no policy update or checkpoint was produced','only_retry_change':'actor.enable_offload false -> true, matching successful v169 resource topology'},'world_model':{'model_version':'track2-v271-endpoint-calibrated-terminal','manifest':str(MANIFEST),'manifest_sha256':sha(MANIFEST),'corrected_capture_gate':str(capture),'corrected_capture_gate_sha256':sha(capture),'long_horizon_gate':str(long),'long_horizon_gate_sha256':sha(long),'runtime_sha256':sha(runtime),'backends_sha256':sha(backends)},'frozen_training':{'initial_policy':'official unmodified Pi0.5 adjust_bottle checkpoint','initial_policy_path':str(official),'initial_policy_sha256':sha(official),'max_steps':5,'trajectories_per_update':256,'total_trajectories':1280,'episode_steps':200,'rollout_steps':200,'rollout_epoch':8,'total_envs':32,'group_size':4,'actor_global_batch_size':6400,'actor_lr':2e-5,'kl_beta':.01,'kl_penalty':'low_var_kl','actor_seed':1471,'env_seed':0,'actor_enable_offload':True,'reference_state_storage':'disk_mmap_during_kl_only','terminal_checkpoint_contents':'full_model_weights_only'},'comparison':{'frozen_best':'v169 step5 = 51/128','v169_schedule':['fresh official initialization','5 effective updates','256 trajectories/update','actor lr 2e-5'],'intended_change':'v271 endpoint-calibrated world model plus small action-normalized KL beta 0.01'},'acceptance_gates':{'action_dim_normalized_approx_kl_abs_max':.01,'action_dim_normalized_clip_fraction_max':.05,'gradient_norm_max':5.0,'all_update_kl_loss_finite_nonnegative':True,'checkpoint_zip_integrity_required':True},'selection_protocol':{'training_audit_only_before_candidate_preregistration':True,'first_public_gate_after_training':'batch00 only','batch01_forbidden_until_batch00_passes':True,'reserved_final128_forbidden_until_unique_candidate_freeze':True},'runner':{'path':str(runner),'sha256':sha(runner)},'guards':{'real_submission':False,'hidden_or_final_data_used_for_training_or_selection':False,'fixed_official_initial_policy':True,'policy_action_injection':False}}
 (REG/'preregistration.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps({'registry':str(REG),'run':str(RUN)},indent=2))
if __name__=='__main__':main()
