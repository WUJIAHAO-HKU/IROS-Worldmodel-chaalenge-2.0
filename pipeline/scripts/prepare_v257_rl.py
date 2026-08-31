#!/usr/bin/env python3
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
B=Path('/root/autodl-tmp/IROS_WAM_2.0 challenge');O=B/'artifacts/strict_track2_official_20260810';J=B/'artifacts/strict_track2_joint_augmentation_20260810';N='v257_v254_conservativekl_h200_r2_step8_lr5e6_beta005_seed1459_20260819';REG=O/'run_registry'/N;RUN=O/'runs'/N;RELEASE=J/'v255_v254_delta_regime_service_seed1457_20260819';LONG=J/'v256_v254_delta_regime_long128_seed1458_20260819'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 if (REG/'preregistration.json').exists() or RUN.exists():raise SystemExit('refusing overwrite')
 capture=RELEASE/'audit/post_grasp_reward_report.json';long=LONG/'audit/long_gate_report.json'
 if json.loads(capture.read_text()).get('passed') is not True or not (RELEASE/'V255_CAPTURE_GATE_PASSED').exists():raise ValueError('v255 capture gate did not pass')
 if json.loads(long.read_text()).get('passed') is not True or not (LONG/'V256_LONG_GATE_PASSED').exists():raise ValueError('v256 long gate did not pass')
 REG.mkdir(parents=True,exist_ok=True);manifest=RELEASE/'release_registration.json';runtime=B/'pipeline/wam_pipeline/v254_delta_regime_terminal_runtime.py';backends=B/'pipeline/wam_pipeline/backends.py';runner=B/'pipeline/scripts/run_strict_track2_conservative_kl.sh'
 x={'format':'strict-track2-v257-v254-conservative-kl-preregistration-v1','timestamp_utc':datetime.now(timezone.utc).isoformat(),'purpose':'fresh fixed-budget conservative GRPO against the capture- and long-horizon-gated v254 world model','run_path':str(RUN),'world_model':{'model_version':'track2-v254-delta-regime-terminal','release':str(RELEASE),'manifest':str(manifest),'manifest_sha256':sha(manifest),'capture_gate':str(capture),'capture_gate_sha256':sha(capture),'long_horizon_gate':str(long),'long_horizon_gate_sha256':sha(long),'runtime_sha256':sha(runtime),'backends_sha256':sha(backends)},'change_from_v237':{'only_intended_algorithmic_change':'v236 causal-close world-model reward -> v254 delta-regime terminal-successor reward','actor_seed':'new deterministic seed 1459','unchanged':['fresh official Pi0.5 initialization','8 updates and 2x200 rollout','actor learning rate 5e-6','KL beta 0.05 and action-normalized low_var_kl','group size 4 and total environments 8']},'frozen_training':{'initial_policy':'official unmodified Pi0.5 adjust_bottle checkpoint','max_steps':8,'episode_steps':200,'rollout_steps':200,'rollout_epoch':2,'total_envs':8,'group_size':4,'actor_global_batch_size':400,'actor_lr':5e-6,'kl_beta':.05,'kl_penalty':'low_var_kl','actor_seed':1459,'env_seed':0,'reference_state_storage':'disk_mmap_during_kl_only','terminal_checkpoint_contents':'full_model_weights_only'},'acceptance_gates':{'action_dim_normalized_approx_kl_abs_max':.01,'action_dim_normalized_clip_fraction_max':.05,'gradient_norm_max':5.,'all_update_kl_loss_finite_nonnegative':True,'checkpoint_zip_integrity_required':True},'public_policy_screen':{'stage_a':'19/32, +2 over baseline, both arms >=50%, grasps not below baseline','stage_b':'75/112, +3 over baseline, both arms >=60%','selection_data':'frozen public unseen train seeds only'},'runner':{'path':str(runner),'sha256':sha(runner)},'guards':{'real_submission':False,'final_128_used_for_training_or_selection':False,'fixed_official_initial_policy':True,'policy_action_injection':False},'prohibited_inputs':['hidden development outcomes','official final outcomes','real contest submission feedback']}
 (REG/'preregistration.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps({'registry':str(REG),'run':str(RUN)},indent=2))
if __name__=='__main__':main()
