#!/usr/bin/env bash
set -euo pipefail
ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'; O="$ROOT/artifacts/strict_track2_official_20260810"; P="$ROOT/pipeline/scripts"
RLINF="$ROOT/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"; OPENPI="$ROOT/third_party/openpi-rlinf-full"; DIFFSYNTH="$O/official_deps/diffsynth_2a2e05f"
ADAPTER="$O/v15_frozen_input_adapter"; OPENPI_CKPT="$ROOT/artifacts/official_resources/pi05_adjust_bottle"; REWARD="$ROOT/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt"; T5="$ROOT/artifacts/official_resources/reward_model/t5-base"
NAME='v386_v169_v385_trainmode_rolloutonly128_seed1497_20260823'; REG="$O/run_registry/$NAME"; RUN="$O/runs/$NAME"
V169="$O/runs/v169_parent_step5_seed1243_20260813/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt"; PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
test -s "$REG/preregistration.json"; test ! -e "$RUN"; mkdir -p "$RUN/audit"
cleanup(){ "$PY" -m ray.scripts.scripts stop --force >/dev/null 2>&1 || true; bash "$P/restart_v385_services.sh" stop >/dev/null 2>&1 || true; bash "$P/restart_v218_services.sh" start >"$REG/restore_v218.log" 2>&1 || true; }
trap cleanup EXIT
TRACK2_SERVICE_REG="$REG" TRACK2_CPUSET=0-5 TRACK2_WAM_RELEASE_CUDA_CACHE=1 bash "$P/restart_v385_services.sh" start >"$REG/restart_v385_before_rollout.log" 2>&1
curl -fsS http://127.0.0.1:18084/health | grep -q '"status":"ready"'
"$PY" -m ray.scripts.scripts stop --force >"$RUN/ray_stop_before.log" 2>&1 || true
taskset -c 6-21 "$PY" -m ray.scripts.scripts start --head --object-store-memory=4294967296 --disable-usage-stats >"$RUN/ray_start.log" 2>&1
cd "$RLINF"
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}" no_proxy="127.0.0.1,localhost,${no_proxy:-}" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 RAYON_NUM_THREADS=2
export TRACK2_ROLLOUT_ONLY_OUTPUT="$RUN/audit/raw_training_mode_rollout.json"
REPO_PATH="$RLINF" EMBODIED_PATH="$RLINF/examples/embodiment" OPENPI_CKPT_PATH="$OPENPI_CKPT" WAN_ROBOTWIN_ADJUST_BOTTLE_CKPT="$ADAPTER" ROBOTWIN_REWARD_MODEL_PATH="$REWARD" T5_MODEL_PATH="$T5" PYTHONPATH="$DIFFSYNTH:$OPENPI/packages/openpi-client/src:$OPENPI/src:$RLINF" PYTHONHASHSEED=0 RAY_DEDUP_LOGS=0 \
taskset -c 6-21 "$PY" "$P/trainmode_rollout_only_embodied_agent.py" \
  --config-path "$RLINF/examples/embodiment/config" --config-name wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05 \
  runner.logger.log_path="$RUN" runner.ckpt_path="$V169" actor.seed=1497 algorithm.rollout_epoch=4 algorithm.group_size=4 \
  env.train.total_num_envs=32 env.train.group_size=4 env.train.seed=0 env.train.max_episode_steps=200 env.train.max_steps_per_rollout_epoch=200 env.train.http.server_url=http://127.0.0.1:18084 \
  env.train.wan_wm_hf_ckpt_path="$ADAPTER" env.train.VAE_path="$ADAPTER/Wan2.2_VAE.pth" env.train.model_path="$ADAPTER/dit_model.safetensors" env.train.initial_image_path="$ADAPTER/dataset/" env.train.enable_kir=false \
  env.train.video_cfg.video_base_dir="$RUN/video/train" env.train.enable_offload=false rollout.enable_offload=false >"$RUN/launcher.log" 2>&1
cd "$ROOT"
"$PY" "$P/audit_v386_rolloutonly.py" --raw "$RUN/audit/raw_training_mode_rollout.json" --preregistration "$REG/preregistration.json" --output "$RUN/audit/rollout_go_no_go.json" >"$RUN/audit/rollout_go_no_go.log" 2>&1
cp "$RUN/audit/rollout_go_no_go.json" "$REG/rollout_go_no_go.json"; cat "$RUN/audit/rollout_go_no_go.log"
