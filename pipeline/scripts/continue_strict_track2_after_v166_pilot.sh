#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRACK2_ROOT:-/root/autodl-tmp/IROS_WAM_2.0 challenge}"
PY="${TRACK2_GO1_PYTHON:-/root/miniconda3/envs/go1/bin/python}"
AUDIT="$ROOT/artifacts/strict_track2_official_20260810"
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
RESOURCE_RUN="$AUDIT/runs/full_budget_complementary_offload_v157_seed1238"
PILOT="$J/v166_singlepass_student_pilot_seed166"
PILOT_PID="$RESOURCE_RUN/audit/v166_student_pilot.pid"
STATE="$RESOURCE_RUN/audit/post_v166_orchestrator.log"
SCREEN_SCRIPT="$ROOT/pipeline/scripts/run_strict_track2_v166_singlepass_student_screen.sh"

exec >> "$STATE" 2>&1
echo "$(date -Iseconds) watcher_started"
while [[ ! -s "$PILOT_PID" ]]; do
  sleep 30
done
pid=$(tr -cd '0-9' < "$PILOT_PID")
while kill -0 "$pid" 2>/dev/null; do
  sleep 30
done

manifest="$PILOT/best/training_manifest.json"
if [[ ! -s "$manifest" ]]; then
  echo "$(date -Iseconds) pilot_failed_no_best_manifest pid=$pid"
  exit 20
fi
"$PY" - "$manifest" <<'PY'
import json, math, sys
doc=json.load(open(sys.argv[1]))
assert doc['format']=='strict-track2-v166-singlepass-visual-student-training-v1'
assert doc['checkpoint_step'] in (50,100,150,200)
assert all(math.isfinite(float(row['ground_truth_validation_mae'])) for row in doc['validation'])
initial=float(doc['initial_ground_truth_validation_mae'])
best=float(doc['best_ground_truth_validation_mae'])
gain=100*(initial-best)/max(abs(initial),1e-12)
assert gain >= 3.0, f'pilot validation gain {gain:.6f}% is below 3%'
counts=doc['source_counts']
assert counts['teacher']+counts['ground_truth']==800
assert counts['left']==counts['right']==400
print(json.dumps({'pilot_gate_passed':True,'validation_improvement_percent':gain,'checkpoint_step':doc['checkpoint_step']}))
PY
echo "$(date -Iseconds) pilot_gate_passed"

nohup env TRACK2_ROOT="$ROOT" "$SCREEN_SCRIPT" \
  > "$RESOURCE_RUN/audit/v166_student_screen_launcher.log" 2>&1 &
screen_pid=$!
printf '%s\n' "$screen_pid" > "$RESOURCE_RUN/audit/v166_student_screen.pid"
echo "$(date -Iseconds) student_screen_started pid=$screen_pid"
