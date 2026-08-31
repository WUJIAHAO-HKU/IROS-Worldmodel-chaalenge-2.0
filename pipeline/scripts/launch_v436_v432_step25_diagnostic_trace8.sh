#!/usr/bin/env bash
# Parent-world-model diagnostic only: 8 trajectories, 200 requests, zero policy updates.
set -euo pipefail
ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge';O="$ROOT/artifacts/strict_track2_official_20260810";P="$ROOT/pipeline/scripts";RLINF="$ROOT/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark";OPENPI="$ROOT/third_party/openpi-rlinf-full";DIFF="$O/official_deps/diffsynth_2a2e05f";ADAPTER="$O/v15_frozen_input_adapter";OPENPI_CKPT="$ROOT/artifacts/official_resources/pi05_adjust_bottle";REWARD="$ROOT/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt";T5="$ROOT/artifacts/official_resources/reward_model/t5-base"
NAME='v436_v432step25_parent_diagnostic_trace8_seed1583_20260823';REG="$O/run_registry/$NAME";RUN="/dev/shm/$NAME";TRACE="$RUN/audit/reward_trace";V169="$O/runs/v169_parent_step5_seed1243_20260813/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt";PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python';GO1='/root/miniconda3/envs/go1/bin/python';RELEASE="$ROOT/artifacts/strict_track2_joint_augmentation_20260810/v436_v432_step25_parent_diagnostic_release";STEP25_GATE="$O/run_registry/v432_public_mirror_prompt_terminal_seed1582_20260823/step25_shortgate.json";STEP25_SHA='dff072aff2f5c64261f9f968cd9ae436440132edbe06a6cf9e1bd2549cdadf7f'
test -s "$RELEASE/v436_diagnostic_manifest.json";test -s "$STEP25_GATE";test "$(sha256sum "$RELEASE/candidate_right/model.pt"|awk '{print $1}')" = "$STEP25_SHA"
"$GO1" - "$RELEASE/v436_diagnostic_manifest.json" "$STEP25_GATE" "$STEP25_SHA" <<'PY'
import json,sys
manifest,gate,expected=sys.argv[1:];m=json.load(open(manifest));g=json.load(open(gate))
assert m["format"]=="track2-v436-v432-step25-diagnostic-release-v1"
assert m["classification"]=="parent world-model diagnostic only" and m["formal_candidate_authorized"] is False
assert m["model_sha256"]["candidate_right"]==expected
assert g["format"]=="strict-track2-v432-step25-shortgate-v1" and g["passed"] is True
assert g["evidence_sha256"]["candidate_model"]==expected
assert g["guards"]["hidden_or_final_data"] is False and g["guards"]["real_submission"] is False and g["guards"]["policy_updates"]==0
PY
test ! -e "$REG";test ! -e "$RUN";mkdir -p "$REG" "$TRACE"
"$GO1" - "$REG/preregistration.json" <<'PY'
import json,sys
json.dump({"format":"strict-track2-v436-v432-step25-trace8-preregistration-v1","classification":"parent world-model diagnostic only","candidate_stage":"passed v432 step25 diagnostic candidate","formal_candidate_authorized":False,"protocol":{"actor_seed":1583,"env_seed":0,"trajectories":8,"requests":200,"group_size":4,"policy_updates":0,"checkpoint_writes":0},"decision":{"learned_request_mean_delta_positive":True,"learned_request_positive_fraction_min":.55,"trajectory_positive_fraction_min":.5,"group_spread_min":.02,"groups_with_spread_min":1,"all_required":True},"guards":{"hidden_or_final_data":False,"real_submission":False,"formal_candidate_authorized":False}},open(sys.argv[1],"w"),indent=2)
PY
cleanup(){ "$PY" -m ray.scripts.scripts stop --force >/dev/null 2>&1||true;TRACK2_SERVICE_REG="$REG" TRACK2_TRACE_DIR="$TRACE" bash "$P/restart_v436_v432_step25_trace_services.sh" stop >/dev/null 2>&1||true;bash "$P/restart_v218_services.sh" start >"$REG/restore_v218.log" 2>&1||true;};trap cleanup EXIT
TRACK2_SERVICE_REG="$REG" TRACK2_TRACE_DIR="$TRACE" TRACK2_CPUSET=0-5 bash "$P/restart_v436_v432_step25_trace_services.sh" start >"$REG/service_start.log" 2>&1
"$PY" -m ray.scripts.scripts stop --force >/dev/null 2>&1||true;taskset -c 6-21 "$PY" -m ray.scripts.scripts start --head --object-store-memory=4294967296 --disable-usage-stats >"$RUN/ray_start.log" 2>&1
cd "$RLINF";export TRACK2_ROLLOUT_ONLY_OUTPUT="$RUN/audit/sequestered_raw.json" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
REPO_PATH="$RLINF" EMBODIED_PATH="$RLINF/examples/embodiment" OPENPI_CKPT_PATH="$OPENPI_CKPT" WAN_ROBOTWIN_ADJUST_BOTTLE_CKPT="$ADAPTER" ROBOTWIN_REWARD_MODEL_PATH="$REWARD" T5_MODEL_PATH="$T5" PYTHONPATH="$DIFF:$OPENPI/packages/openpi-client/src:$OPENPI/src:$RLINF" taskset -c 6-21 "$PY" "$P/trainmode_rollout_only_embodied_agent.py" --config-path "$RLINF/examples/embodiment/config" --config-name wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05 runner.logger.log_path="$RUN" runner.ckpt_path="$V169" actor.seed=1583 algorithm.rollout_epoch=1 algorithm.group_size=4 env.train.total_num_envs=8 env.train.group_size=4 env.train.seed=0 env.train.max_episode_steps=200 env.train.max_steps_per_rollout_epoch=200 env.train.http.server_url=http://127.0.0.1:18084 env.train.wan_wm_hf_ckpt_path="$ADAPTER" env.train.VAE_path="$ADAPTER/Wan2.2_VAE.pth" env.train.model_path="$ADAPTER/dit_model.safetensors" env.train.initial_image_path="$ADAPTER/dataset/" env.train.enable_kir=false env.train.video_cfg.video_base_dir="$RUN/video/train" env.train.enable_offload=false rollout.enable_offload=true >"$RUN/launcher.log" 2>&1
"$PY" -m ray.scripts.scripts stop --force >/dev/null 2>&1||true;TRACK2_SERVICE_REG="$REG" TRACK2_TRACE_DIR="$TRACE" bash "$P/restart_v436_v432_step25_trace_services.sh" stop >/dev/null 2>&1||true
cd "$ROOT";set +e;PYTHONPATH="$ROOT/pipeline:$RLINF" taskset -c 0-11 "$GO1" "$P/analyze_v436_v432_step25_trace8.py" --trace-dir "$TRACE" --preregistration "$REG/preregistration.json" --reward-checkpoint "$REWARD" --t5-model "$T5" --output "$REG/result.json" --device cuda --batch-size 32 >"$REG/analyzer.log" 2>&1;status=$?;set -e;test -s "$REG/result.json";exit "$status"
