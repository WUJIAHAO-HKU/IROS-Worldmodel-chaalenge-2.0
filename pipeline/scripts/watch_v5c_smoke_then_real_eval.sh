#!/usr/bin/env bash
set -euo pipefail

# Wait for the official one-step RL smoke checkpoint, then run the complete
# 128-trajectory real RoboTwin evaluation (baseline + candidate).  This is a
# convenience watcher only; it does not alter the official policy/reward/RL
# configuration.
ROOT="${TRACK2_ROOT:-/root/autodl-tmp/IROS_WAM_2.0 challenge}"
RUN="$ROOT/artifacts/strict_track2_joint_augmentation_20260810/recursive_arm_routed_v5c_alpha020/rl_smoke"
CKPT="$RUN/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt"
EVAL_ROOT="$ROOT/artifacts/strict_track2_official_20260810/real_robotwin_eval"
OUT="$EVAL_ROOT/instrumented_metrics_v5c_smoke_step1"
LOG="$EVAL_ROOT/v5c_smoke_eval_watch.log"
PY="/root/autodl-tmp/conda_envs/rlinf_track2/bin/python"

printf '%s waiting_for_checkpoint=%s\n' "$(date -Iseconds)" "$CKPT" >> "$LOG"
while [[ ! -s "$CKPT" ]]; do
  sleep 30
done
printf '%s checkpoint_ready=%s size=%s\n' "$(date -Iseconds)" "$CKPT" "$(stat -c %s "$CKPT")" >> "$LOG"

# Stop only this smoke run and its two local transport processes before the
# real-environment evaluator claims the GPU.
parent_pid="$(pgrep -f 'bash /root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/run_v5c_official_rl_smoke.sh' | head -1 || true)"
if [[ -n "$parent_pid" ]]; then
  kill "$parent_pid" 2>/dev/null || true
  sleep 15
fi
for pattern in \
  'train_embodied_agent.py.*rl_smoke' \
  'wam_pipeline.rlinf_bridge.server.*--port 18080' \
  'pipeline/scripts/serve.py'; do
  while read -r pid; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done < <(pgrep -f "$pattern" || true)
done
sleep 10
"$PY" -m ray.scripts.scripts stop --force >"$EVAL_ROOT/v5c_smoke_eval_ray_stop.log" 2>&1 || true

export TRACK2_FINAL128_CHECKPOINT="$CKPT"
export TRACK2_FINAL128_VARIANT="v5c_smoke_step1"
export TRACK2_INSTRUMENTED_OUTPUT_ROOT="$OUT"
mkdir -p "$OUT"
bash "$ROOT/pipeline/scripts/run_strict_track2_final128_eval.sh" \
  >"$OUT/launcher.log" 2>&1
printf '%s real_eval_complete=%s\n' "$(date -Iseconds)" "$OUT" >> "$LOG"
