#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  printf 'Usage: %s BATCH_ID ENV_COUNT\n' "$0" >&2
  exit 2
fi
BATCH_ID="$1"
ENV_COUNT="$2"
[[ "$BATCH_ID" =~ ^[0-9][0-9]$ ]] || exit 2
[[ "$ENV_COUNT" =~ ^[1-9][0-9]*$ ]] || exit 2

ROOT="/root/autodl-tmp/IROS_WAM_2.0 challenge"
RLINF_ROOT="$ROOT/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
ROBOTWIN_ROOT="$ROOT/artifacts/strict_track2_official_20260810/official_deps/RoboTwin_RLinf_support"
OPENPI_ROOT="$ROOT/third_party/openpi-rlinf-full"
AUDIT_ROOT="$ROOT/artifacts/strict_track2_official_20260810"
CAPTURE_RUNTIME="$AUDIT_ROOT/real_robotwin_eval/onpolicy_capture_runtime"
JOINT_ROOT="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
SEED_PATH="$JOINT_ROOT/onpolicy_train_seeds128/batch_$BATCH_ID.json"
RUN_ROOT="$JOINT_ROOT/onpolicy_instruction_probe/batch_$BATCH_ID/eval"
CAPTURE_ROOT="$JOINT_ROOT/onpolicy_instruction_probe/batch_$BATCH_ID/chunks"
LOG_PATH="$RUN_ROOT/launcher.log"
MODEL_ROOT="$ROOT/artifacts/official_resources/pi05_adjust_bottle"
PYTHON_BIN="/root/autodl-tmp/conda_envs/rlinf_track2/bin/python"
ACTOR_SEED=$((20260810 + 10#$BATCH_ID))

test -s "$SEED_PATH"
grep -q task_instruction "$CAPTURE_RUNTIME/robotwin/envs/vector_env.py"
if [[ -f "$RUN_ROOT/complete.json" ]]; then
  printf 'SKIP batch=%s reason=complete\n' "$BATCH_ID"
  exit 0
fi
if [[ -e "$RUN_ROOT" || -e "$CAPTURE_ROOT" ]]; then
  printf 'Refusing to overwrite partial instruction probe %s\n' "$BATCH_ID" >&2
  exit 3
fi
if pgrep -af '[e]val_embodied_agent.py|[t]rain_embodied_agent.py' >/dev/null; then
  printf 'Refusing to contend with active official RL job\n' >&2
  exit 4
fi

"$PYTHON_BIN" -m ray.scripts.scripts stop --force >"/tmp/ray_stop_instruction_${BATCH_ID}.log" 2>&1 || true
mkdir -p "$RUN_ROOT" "$CAPTURE_ROOT" /tmp/xdg-robotwin-instruction
cd /tmp
REPO_PATH="$RLINF_ROOT" \
EMBODIED_PATH="$RLINF_ROOT/examples/embodiment" \
ROBOTWIN_CAPTURE_ROOT="$CAPTURE_ROOT" \
PYTHONPATH="$CAPTURE_RUNTIME:$OPENPI_ROOT/packages/openpi-client/src:$OPENPI_ROOT/src:$ROBOTWIN_ROOT:$RLINF_ROOT" \
PYTHONHASHSEED=0 \
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
XDG_RUNTIME_DIR=/tmp/xdg-robotwin-instruction \
RAY_DEDUP_LOGS=0 \
"$PYTHON_BIN" "$RLINF_ROOT/examples/embodiment/eval_embodied_agent.py" \
  --config-path "$AUDIT_ROOT/real_robotwin_eval/runtime_config_tree" \
  --config-name robotwin_adjust_bottle_ppo_openpi_pi05_eval \
  runner.logger.log_path="$RUN_ROOT" \
  runner.logger.experiment_name="instruction-probe-batch-$BATCH_ID" \
  env.eval.total_num_envs="$ENV_COUNT" \
  env.eval.max_episode_steps=8 \
  env.eval.assets_path="$ROBOTWIN_ROOT" \
  env.eval.seeds_path="$SEED_PATH" \
  env.eval.video_cfg.save_video=false \
  actor.seed="$ACTOR_SEED" \
  actor.model.model_path="$MODEL_ROOT" \
  actor.model.num_action_chunks=8 \
  actor.model.action_dim=14 \
  actor.model.add_value_head=true \
  actor.model.openpi.config_name=pi05_aloha_robotwin_head_adjust_bottle \
  actor.model.openpi.num_images_in_input=1 \
  actor.model.openpi.action_chunk=8 \
  actor.model.openpi.action_env_dim=14 \
  actor.model.openpi.noise_level=0.3 \
  runner.ckpt_path=null >"$LOG_PATH" 2>&1 &
EVALUATOR_PID=$!
deadline=$((SECONDS + 600))
while kill -0 "$EVALUATOR_PID" 2>/dev/null; do
  chunk_files="$(find "$CAPTURE_ROOT" -type f -name 'chunk_000.npz' | wc -l)"
  if [[ "$chunk_files" -eq "$ENV_COUNT" ]]; then
    # Wait for atomic renames and filesystem metadata to settle before stopping
    # the evaluation framework, which otherwise spends minutes closing Ray.
    sleep 3
    break
  fi
  if (( SECONDS >= deadline )); then
    kill -TERM "$EVALUATOR_PID" 2>/dev/null || true
    wait "$EVALUATOR_PID" || true
    printf 'Instruction probe timed out: chunks=%s expected=%s\n' "$chunk_files" "$ENV_COUNT" >&2
    exit 5
  fi
  sleep 2
done
kill -INT "$EVALUATOR_PID" 2>/dev/null || true
for _ in {1..10}; do
  kill -0 "$EVALUATOR_PID" 2>/dev/null || break
  sleep 1
done
kill -TERM "$EVALUATOR_PID" 2>/dev/null || true
for _ in {1..5}; do
  kill -0 "$EVALUATOR_PID" 2>/dev/null || break
  sleep 1
done
kill -KILL "$EVALUATOR_PID" 2>/dev/null || true
wait "$EVALUATOR_PID" || true
"$PYTHON_BIN" -m ray.scripts.scripts stop --force >"/tmp/ray_stop_instruction_after_${BATCH_ID}.log" 2>&1 || true

chunk_files="$(find "$CAPTURE_ROOT" -type f -name 'chunk_000.npz' | wc -l)"
if [[ "$chunk_files" -ne "$ENV_COUNT" ]]; then
  printf 'Instruction probe cardinality mismatch: chunks=%s expected=%s\n' "$chunk_files" "$ENV_COUNT" >&2
  exit 5
fi
"$PYTHON_BIN" - "$CAPTURE_ROOT" "$RUN_ROOT/complete.json" <<'PY'
import json, sys
from pathlib import Path
import numpy as np

root, output = map(Path, sys.argv[1:])
rows = []
for path in sorted(root.glob("seed_*/chunk_000.npz"), key=lambda p: int(p.parent.name[5:])):
    with np.load(path, allow_pickle=False) as values:
        rows.append({
            "seed": int(values["seed"]),
            "arm_right": bool(values["arm_right"]),
            "task_instruction": str(values["task_instruction"]),
        })
output.write_text(json.dumps({"format": "strict-track2-instruction-probe-v1", "rows": rows}, indent=2) + "\n")
print(json.dumps({"batch": output.parent.parent.name, "instructions": len(rows)}))
PY
