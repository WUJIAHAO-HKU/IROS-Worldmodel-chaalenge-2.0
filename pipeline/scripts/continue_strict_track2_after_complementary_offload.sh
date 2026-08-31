#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRACK2_ROOT:-/root/autodl-tmp/IROS_WAM_2.0 challenge}"
AUDIT="$ROOT/artifacts/strict_track2_official_20260810"
RUN="$AUDIT/runs/full_budget_complementary_offload_v157_seed1238"
BRIDGE="$AUDIT/v157_complementary_offload_bridge/bridge.log"
STATE="$RUN/audit/post_completion_orchestrator.log"
SUCCESS="$RUN/audit/gpu_after_completion.csv"
STUDENT_SCRIPT="$ROOT/pipeline/scripts/run_strict_track2_v166_singlepass_student_pilot.sh"
STUDENT_PID="$RUN/audit/v166_student_pilot.pid"

mkdir -p "$RUN/audit"
exec >> "$STATE" 2>&1
echo "$(date -Iseconds) watcher_started"

while [[ ! -e "$SUCCESS" ]]; do
  if ! pgrep -af 'run_strict_track2_full_budget_complementary_offload_smoke.sh' >/dev/null; then
    echo "$(date -Iseconds) diagnostic_ended_without_success_marker"
    exit 10
  fi
  sleep 30
done

chunks=$(grep -c 'POST /chunk_step' "$BRIDGE" || true)
fatal=$(grep -Eic 'CUDA out of memory|OutOfMemoryError|RayTaskError|Traceback \(most recent call last\)' "$RUN/launcher.log" || true)
if [[ "$chunks" -ne 200 || "$fatal" -ne 0 ]]; then
  echo "$(date -Iseconds) diagnostic_gate_failed chunks=$chunks fatal=$fatal"
  exit 11
fi
CHECKPOINT="$RUN/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_1/actor"
if [[ ! -d "$CHECKPOINT" ]] || ! find "$CHECKPOINT" -type f -print -quit | grep -q .; then
  echo "$(date -Iseconds) diagnostic_gate_failed missing_global_step_1_checkpoint=$CHECKPOINT"
  exit 13
fi
if ! grep -q 'Saving checkpoint at step 1' "$RUN/launcher.log"; then
  echo "$(date -Iseconds) diagnostic_gate_failed checkpoint_log_marker_missing"
  exit 14
fi
echo "$(date -Iseconds) diagnostic_gate_passed chunks=$chunks fatal=$fatal"

if [[ -s "$RUN/audit/gpu_sampler.pid" ]]; then
  sampler=$(tr -cd '0-9' < "$RUN/audit/gpu_sampler.pid")
  if [[ -n "$sampler" && -r "/proc/$sampler/cmdline" ]] \
    && tr '\0' ' ' < "/proc/$sampler/cmdline" | grep -q 'nvidia-smi.*query-compute-apps'; then
    kill "$sampler" || true
    echo "$(date -Iseconds) stopped_gpu_sampler pid=$sampler"
  fi
fi
sleep 1
RESULT="$RUN/audit/strict_track2_v157_complementary_offload_result.json"
/root/miniconda3/envs/go1/bin/python \
  "$ROOT/pipeline/scripts/summarize_strict_track2_complementary_offload.py" \
  --run "$RUN" \
  --bridge-log "$BRIDGE" \
  --output "$RESULT" \
  > "$RUN/audit/strict_track2_v157_complementary_offload_result.log" 2>&1
/root/miniconda3/envs/go1/bin/python - "$RESULT" <<'PY'
import json,sys
doc=json.load(open(sys.argv[1]))
assert doc['passed'] is True, doc['failed_checks']
PY

/root/autodl-tmp/conda_envs/rlinf_track2/bin/python -m ray.scripts.scripts stop --force \
  > "$RUN/ray_stop_after.log" 2>&1 || true

MIXED_CACHE="$ROOT/artifacts/strict_track2_joint_augmentation_20260810/v157_balanced_mixed_visual_cache64.npz"
MIXED_MANIFEST="$ROOT/artifacts/strict_track2_joint_augmentation_20260810/v157_balanced_mixed_visual_cache64.manifest.json"
if [[ -e "$MIXED_CACHE" && ! -e "$MIXED_MANIFEST" ]] \
  || [[ ! -e "$MIXED_CACHE" && -e "$MIXED_MANIFEST" ]]; then
  echo "mixed visual cache is partial; refusing to continue" >&2
  exit 12
fi
if [[ ! -e "$MIXED_CACHE" && ! -e "$MIXED_MANIFEST" ]]; then
  cd "$ROOT"
  PYTHONPATH="$ROOT/pipeline" /root/miniconda3/envs/go1/bin/python \
    pipeline/scripts/build_strict_track2_v157_mixed_visual_cache.py \
    --preregistration "$ROOT/pipeline/config/strict_track2_v157_mixed_visual_cache_preregistration.json" \
    --windows "$ROOT/artifacts/strict_track2_joint_augmentation_20260810/formal_joint_parent_windows_full128" \
    --source-manifest "$ROOT/artifacts/strict_track2_joint_augmentation_20260810/formal_joint_parent_windows_full128/window_sources.json" \
    --synthetic-cache "$ROOT/artifacts/strict_track2_joint_augmentation_20260810/v157_exact_reward_cache64.npz" \
    --api-url http://127.0.0.1:8001 \
    --token local-dev-token \
    --model-version track2-v15.7-hybrid-action-gated-reward-safe-blend12 \
    --output "$MIXED_CACHE" \
    --manifest "$MIXED_MANIFEST" \
    > "$RUN/audit/v157_mixed_visual_cache_builder.log" 2>&1
  echo "$(date -Iseconds) mixed_visual_cache_built"
fi
test -s "$MIXED_CACHE"
test -s "$MIXED_MANIFEST"

for port in 18080 8001; do
  pid=$(ss -ltnp "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | head -1)
  if [[ -z "$pid" || ! -r "/proc/$pid/cmdline" ]]; then
    continue
  fi
  command=$(tr '\0' ' ' < "/proc/$pid/cmdline")
  if [[ "$port" == 18080 && "$command" == *wam_pipeline.rlinf_bridge.server* ]] \
    || [[ "$port" == 8001 && "$command" == *pipeline/scripts/serve.py* ]]; then
    kill "$pid" || true
    echo "$(date -Iseconds) stopped_listener port=$port pid=$pid"
  fi
done

sleep 5
nohup env TRACK2_ROOT="$ROOT" "$STUDENT_SCRIPT" \
  > "$RUN/audit/v166_student_pilot_launcher.log" 2>&1 &
student_pid=$!
printf '%s\n' "$student_pid" > "$STUDENT_PID"
echo "$(date -Iseconds) student_pilot_started pid=$student_pid"
