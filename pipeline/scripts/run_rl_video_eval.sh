#!/usr/bin/env bash
set -u
ROOT=/tmp/iros_track2
RLINF_ROOT="$ROOT/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
OPENPI_ROOT="$ROOT/third_party/openpi-rlinf-full"
ROBOTWIN_ROOT="$ROOT/artifacts/strict_track2_official_20260810/official_deps/RoboTwin_RLinf_support"
INSTRUMENTED_ROOT="$ROOT/artifacts/strict_track2_official_20260810/real_robotwin_eval/instrumented_runtime"
PY=/root/autodl-tmp/conda_envs/rlinf_track2/bin/python
OUT="$ROOT/artifacts/strict_track2_official_20260810/video_exports/$1"
SEEDS="$ROOT/artifacts/strict_track2_official_20260810/video_exports/video_seed_$1.json"
mkdir -p "$OUT"
cd /tmp
REPO_PATH="$RLINF_ROOT" EMBODIED_PATH="$RLINF_ROOT/examples/embodiment" \
PYTHONPATH="$INSTRUMENTED_ROOT:$OPENPI_ROOT/packages/openpi-client/src:$OPENPI_ROOT/src:$ROBOTWIN_ROOT:$RLINF_ROOT" \
PYTHONHASHSEED=0 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json XDG_RUNTIME_DIR=/tmp/xdg-robotwin RAY_DEDUP_LOGS=0 \
TRACK2_ACTION_CAPTURE_DIR="$OUT/actions" \
"$PY" "$RLINF_ROOT/examples/embodiment/eval_embodied_agent.py" \
  --config-path "$ROOT/artifacts/strict_track2_official_20260810/real_robotwin_eval/runtime_config_tree" \
  --config-name robotwin_adjust_bottle_ppo_openpi_pi05_eval \
  runner.logger.log_path="$OUT" runner.logger.experiment_name="rl-$1" \
  env.eval.total_num_envs=1 env.eval.assets_path="$ROBOTWIN_ROOT" env.eval.seeds_path="$SEEDS" \
  env.eval.video_cfg.save_video=true env.eval.video_cfg.video_base_dir="$OUT/video" \
  actor.model.model_path="$ROOT/artifacts/official_resources/pi05_adjust_bottle" \
  actor.model.num_action_chunks=8 actor.model.action_dim=14 actor.model.add_value_head=true \
  actor.model.openpi.config_name=pi05_aloha_robotwin_head_adjust_bottle actor.model.openpi.num_images_in_input=1 \
  actor.model.openpi.action_chunk=8 actor.model.openpi.action_env_dim=14 actor.model.openpi.noise_level=0.3 \
  runner.ckpt_path="$ROOT/artifacts/strict_track2_official_20260810/runs/probability_consistent_four_step_v169_seed1243_retry2/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_4/actor/model_state_dict/full_weights.pt" \
  > "$OUT/launcher.log" 2>&1
