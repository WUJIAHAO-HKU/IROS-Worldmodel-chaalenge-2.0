#!/usr/bin/env bash
# Prepared only. Diagnostic continuation evidence; never a formal/final candidate selector.
set -euo pipefail

ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
O="$ROOT/artifacts/strict_track2_official_20260810"
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
P="$ROOT/pipeline/scripts"
RLINF="$ROOT/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
OPENPI="$ROOT/third_party/openpi-rlinf-full"
DIFFSYNTH="$O/official_deps/diffsynth_2a2e05f"
ADAPTER="$O/v15_frozen_input_adapter"
OPENPI_CKPT="$ROOT/artifacts/official_resources/pi05_adjust_bottle"
REWARD="$ROOT/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt"
T5="$ROOT/artifacts/official_resources/reward_model/t5-base"
RELEASE="$J/v169_instruction_arm_routed_release"
NAME='v431r1_v169_originalwm_diagnostic_trace32_seed1580_20260823'
REG="$O/run_registry/$NAME"
RUN="/dev/shm/$NAME"
TRACE="$RUN/audit/reward_trace"
V169="$O/runs/v169_parent_step5_seed1243_20260813/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
GO1='/root/miniconda3/envs/go1/bin/python'

test ! -e "$REG"
test ! -e "$RUN"
test "$(sha256sum "$V169" | awk '{print $1}')" = '41f0bee86472d22cbb6ab6a7d0060f08f0ba0aad93d260582a5f7f078bca776b'
mkdir -p "$REG" "$TRACE"

"$GO1" - "$ROOT" "$REG/preregistration.json" <<'PY'
import datetime,hashlib,json,pathlib,sys
root,output=map(pathlib.Path,sys.argv[1:])
o=root/'artifacts/strict_track2_official_20260810'
j=root/'artifacts/strict_track2_joint_augmentation_20260810'
paths={
 'runtime':root/'pipeline/wam_pipeline/v431_v169_reward_trace_runtime.py',
 'v169_runtime':root/'pipeline/wam_pipeline/v169_arm_routed_runtime.py',
 'backends':root/'pipeline/wam_pipeline/backends.py',
 'analyzer':root/'pipeline/scripts/analyze_v431_v169_full_trajectory_reward_trace.py',
 'service':root/'pipeline/scripts/restart_v431_v169_reward_trace_services.sh',
 'launcher':root/'pipeline/scripts/launch_v431_v169_originalwm_diagnostic_reward_trace32.sh',
 'release_manifest':j/'v169_instruction_arm_routed_release/v169_arm_routed_manifest.json',
 'cache_replay':j/'v169_instruction_arm_routed_release/cache_replay_report.json',
 'service_acceptance':o/'service_acceptance/v169_protocol_acceptance.json',
 'v169_policy':o/'runs/v169_parent_step5_seed1243_20260813/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt',
}
for path in paths.values():
    if not path.is_file(): raise FileNotFoundError(path)
if not json.loads(paths['cache_replay'].read_text()).get('passed'):
    raise RuntimeError('v169 cache replay failed')
if not json.loads(paths['service_acceptance'].read_text()).get('passed'):
    raise RuntimeError('v169 service acceptance failed')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if sha(paths['v169_policy'])!='41f0bee86472d22cbb6ab6a7d0060f08f0ba0aad93d260582a5f7f078bca776b':
    raise RuntimeError('v169 policy integrity failed')
payload={
 'format':'strict-track2-v431-v169-diagnostic-full-trajectory-reward-trace32-preregistration-v1',
 'registered_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'classification':'nonformal diagnostic only',
 'purpose':'Fresh-seed, zero-update audit of raw within-group reward rankability for the original v169 world-model chain and frozen v169-step5 continuation policy.',
 'world_model':{'model_version':'track2-v16.9-instruction-arm-routed','backend':'v169-arm-routed'},
 'protocol':{
   'actor_seed':1580,'env_seed':0,'trajectories':32,'requests':800,
   'steps_per_trajectory':25,'episode_steps':200,'group_size':4,
   'policy_updates':0,'checkpoint_writes':0,'trajectory_discount':.99,
 },
 'decision':{
   'mean_trajectory_return_min':.06,
   'group_return_spread_min':.02,
   'groups_with_spread_min':4,
   'right_routed_requests_min':200,
   'all_required':True,
   'on_fail':'Reject the optional v430 diagnostic continuation.',
   'on_pass':'Permit only the separately prepared one-update v430 nonformal diagnostic; never authorize formal/public/final selection.',
 },
 'formal_contract':{
   'formal_candidate_authorized':False,
   'reason':'v430 continues v169 step5 and changes LR/KL; formal Track2 requires raw Pi0.5 fresh training with unmodified official fixed-budget config after parent freeze.',
 },
 'evidence_sha256':{key:sha(value) for key,value in paths.items()},
 'guards':{
   'v169_rgb_output_unchanged_by_trace':True,'policy_updates':0,'checkpoint_writes':0,
   'simulator_outcomes_used':False,'hidden_or_final_data':False,'real_submission':False,
 },
}
output.write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps(payload,indent=2))
PY

cleanup() {
  "$PY" -m ray.scripts.scripts stop --force >/dev/null 2>&1 || true
  TRACK2_SERVICE_REG="$REG" TRACK2_TRACE_DIR="$TRACE" \
    bash "$P/restart_v431_v169_reward_trace_services.sh" stop >/dev/null 2>&1 || true
  bash "$P/restart_v218_services.sh" start >"$REG/restore_v218.log" 2>&1 || true
}
trap cleanup EXIT

TRACK2_SERVICE_REG="$REG" TRACK2_TRACE_DIR="$TRACE" TRACK2_CPUSET=0-5 \
  bash "$P/restart_v431_v169_reward_trace_services.sh" start \
  >"$REG/restart_trace_before_rollout.log" 2>&1
"$PY" -m ray.scripts.scripts stop --force >"$RUN/ray_stop_before.log" 2>&1 || true
taskset -c 6-21 "$PY" -m ray.scripts.scripts start --head \
  --object-store-memory=4294967296 --disable-usage-stats >"$RUN/ray_start.log" 2>&1

cd "$RLINF"
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 RAYON_NUM_THREADS=2
export TRACK2_ROLLOUT_ONLY_OUTPUT="$RUN/audit/sequestered_raw.json"
REPO_PATH="$RLINF" EMBODIED_PATH="$RLINF/examples/embodiment" \
OPENPI_CKPT_PATH="$OPENPI_CKPT" WAN_ROBOTWIN_ADJUST_BOTTLE_CKPT="$ADAPTER" \
ROBOTWIN_REWARD_MODEL_PATH="$REWARD" T5_MODEL_PATH="$T5" \
PYTHONPATH="$DIFFSYNTH:$OPENPI/packages/openpi-client/src:$OPENPI/src:$RLINF" \
PYTHONHASHSEED=0 RAY_DEDUP_LOGS=0 taskset -c 6-21 "$PY" \
  "$P/trainmode_rollout_only_embodied_agent.py" \
  --config-path "$RLINF/examples/embodiment/config" \
  --config-name wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05 \
  runner.logger.log_path="$RUN" runner.ckpt_path="$V169" actor.seed=1580 \
  algorithm.rollout_epoch=1 algorithm.group_size=4 \
  env.train.total_num_envs=32 env.train.group_size=4 env.train.seed=0 \
  env.train.max_episode_steps=200 env.train.max_steps_per_rollout_epoch=200 \
  env.train.http.server_url=http://127.0.0.1:18084 \
  env.train.wan_wm_hf_ckpt_path="$ADAPTER" env.train.VAE_path="$ADAPTER/Wan2.2_VAE.pth" \
  env.train.model_path="$ADAPTER/dit_model.safetensors" \
  env.train.initial_image_path="$ADAPTER/dataset/" env.train.enable_kir=false \
  env.train.video_cfg.video_base_dir="$RUN/video/train" env.train.enable_offload=false \
  rollout.enable_offload=true >"$RUN/launcher.log" 2>&1

"$PY" -m ray.scripts.scripts stop --force >"$RUN/ray_stop_before_reward.log" 2>&1 || true
TRACK2_SERVICE_REG="$REG" TRACK2_TRACE_DIR="$TRACE" \
  bash "$P/restart_v431_v169_reward_trace_services.sh" stop >/dev/null 2>&1 || true
cd "$ROOT"
set +e
PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts:$RLINF" OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 \
  taskset -c 0-11 "$GO1" "$P/analyze_v431_v169_full_trajectory_reward_trace.py" \
  --trace-dir "$TRACE" --preregistration "$REG/preregistration.json" \
  --reward-checkpoint "$REWARD" --t5-model "$T5" \
  --output "$RUN/audit/reward_signal_analysis.json" --device cuda --batch-size 32 \
  >"$RUN/audit/reward_signal_analysis.log" 2>&1
status=$?
set -e
test -s "$RUN/audit/reward_signal_analysis.json"
cp "$RUN/audit/reward_signal_analysis.json" "$REG/reward_signal_analysis.json"
cp "$RUN/audit/reward_signal_analysis.log" "$REG/reward_signal_analysis.log"
cat "$REG/reward_signal_analysis.log"
exit "$status"
