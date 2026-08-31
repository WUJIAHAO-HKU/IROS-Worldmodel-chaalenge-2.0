#!/usr/bin/env bash
# Fresh-seed absolute official-reward diagnostic of the unchanged v169 chain.
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
NAME='v438_v169_absolute_signal_trace8_seed1585_20260823'
REG="$O/run_registry/$NAME"
RUN="/dev/shm/$NAME"
TRACE="$RUN/audit/reward_trace"
V169="$O/runs/v169_parent_step5_seed1243_20260813/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
GO1='/root/miniconda3/envs/go1/bin/python'

test ! -e "$REG"
test ! -e "$RUN"
test -s "$RELEASE/v169_arm_routed_manifest.json"
test -s "$V169"
test "$(sha256sum "$V169" | awk '{print $1}')" = \
  '41f0bee86472d22cbb6ab6a7d0060f08f0ba0aad93d260582a5f7f078bca776b'
mkdir -p "$REG" "$TRACE"

"$GO1" - "$REG/preregistration.json" <<'PY'
import json,sys
payload={
  "format":"strict-track2-v438-v169-absolute-signal-trace8-preregistration-v1",
  "classification":"nonformal diagnostic only",
  "formal_candidate_authorized":False,
  "purpose":"Fresh-seed absolute official-reward rankability test of the unchanged original v169 world-model backend and frozen v169-step5 policy.",
  "world_model":{"model_version":"track2-v16.9-instruction-arm-routed","backend":"v431-v169-reward-trace","model":"original frozen v169"},
  "protocol":{"actor_seed":1585,"env_seed":0,"trajectories":8,"requests":200,"steps_per_trajectory":25,"episode_steps":200,"group_size":4,"policy_updates":0,"checkpoint_writes":0},
  "decision":{"trajectory_discount":0.99,"mean_absolute_return_min":0.06,"absolute_group_spread_min":0.02,"groups_with_spread_min":1,"all_required":True,"on_fail":"Reject v169 absolute signal as insufficient for further diagnostic continuation.","on_pass":"Diagnostic evidence only; no policy update or formal candidate is authorized."},
  "guards":{"opponent_or_delta_required":False,"policy_updates":0,"checkpoint_writes":0,"simulator_outcomes_used":False,"hidden_or_final_data":False,"real_submission":False},
}
json.dump(payload,open(sys.argv[1],"w"),indent=2)
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
  >"$REG/service_start.log" 2>&1
"$PY" -m ray.scripts.scripts stop --force >/dev/null 2>&1 || true
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
  runner.logger.log_path="$RUN" runner.ckpt_path="$V169" actor.seed=1585 \
  algorithm.rollout_epoch=1 algorithm.group_size=4 \
  env.train.total_num_envs=8 env.train.group_size=4 env.train.seed=0 \
  env.train.max_episode_steps=200 env.train.max_steps_per_rollout_epoch=200 \
  env.train.http.server_url=http://127.0.0.1:18084 \
  env.train.wan_wm_hf_ckpt_path="$ADAPTER" \
  env.train.VAE_path="$ADAPTER/Wan2.2_VAE.pth" \
  env.train.model_path="$ADAPTER/dit_model.safetensors" \
  env.train.initial_image_path="$ADAPTER/dataset/" env.train.enable_kir=false \
  env.train.video_cfg.video_base_dir="$RUN/video/train" \
  env.train.enable_offload=false rollout.enable_offload=true \
  >"$RUN/launcher.log" 2>&1

"$PY" -m ray.scripts.scripts stop --force >/dev/null 2>&1 || true
TRACK2_SERVICE_REG="$REG" TRACK2_TRACE_DIR="$TRACE" \
  bash "$P/restart_v431_v169_reward_trace_services.sh" stop >/dev/null 2>&1 || true
cd "$ROOT"
set +e
PYTHONPATH="$ROOT/pipeline:$RLINF" taskset -c 0-11 "$GO1" \
  "$P/analyze_v438_v169_absolute_signal_trace8.py" \
  --trace-dir "$TRACE" --preregistration "$REG/preregistration.json" \
  --reward-checkpoint "$REWARD" --t5-model "$T5" \
  --output "$REG/result.json" --device cuda --batch-size 32 \
  >"$REG/analyzer.log" 2>&1
status=$?
set -e
test -s "$REG/result.json"
exit "$status"
