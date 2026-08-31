#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRACK2_ROOT:-/root/autodl-tmp/IROS_WAM_2.0 challenge}"
PY="${TRACK2_GO1_PYTHON:-/root/miniconda3/envs/go1/bin/python}"
AUDIT="$ROOT/artifacts/strict_track2_official_20260810"
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
RESOURCE_RUN="$AUDIT/runs/full_budget_complementary_offload_v157_seed1238"
SCREEN_PID="$RESOURCE_RUN/audit/v166_student_screen.pid"
SCREEN="$J/v166_singlepass_student_pilot_seed166/screen/v166_offline_screen.json"
STATE="$RESOURCE_RUN/audit/post_v166_screen_orchestrator.log"
ONE_UPDATE="$ROOT/pipeline/scripts/run_strict_track2_v166_one_update.sh"
MIXED_SCREEN_SCRIPT="$ROOT/pipeline/scripts/run_strict_track2_v166_mixed_visual_screen.sh"
MIXED_SCREEN="$J/v166_singlepass_student_pilot_seed166/screen/v166_mixed_visual_screen.json"

exec >> "$STATE" 2>&1
echo "$(date -Iseconds) watcher_started"
while [[ ! -s "$SCREEN_PID" ]]; do
  sleep 30
done
pid=$(tr -cd '0-9' < "$SCREEN_PID")
while kill -0 "$pid" 2>/dev/null; do
  sleep 30
done
if [[ ! -s "$SCREEN" ]]; then
  echo "$(date -Iseconds) screen_failed_no_result pid=$pid"
  exit 30
fi
"$PY" - "$SCREEN" <<'PY'
import json,sys
doc=json.load(open(sys.argv[1]))
assert doc['format']=='strict-track2-v166-singlepass-student-offline-screen-v1'
assert doc['passed'] is True, doc['failed_checks']
print(json.dumps({'offline_screen_passed':True,'candidate':doc['candidate']}))
PY
echo "$(date -Iseconds) offline_screen_gate_passed"

env TRACK2_ROOT="$ROOT" "$MIXED_SCREEN_SCRIPT"
"$PY" - "$MIXED_SCREEN" <<'PY'
import json,sys
doc=json.load(open(sys.argv[1]))
assert doc['format']=='strict-track2-v166-balanced-mixed-visual-screen-v1'
assert doc['passed'] is True, doc['failed_checks']
print(json.dumps({'mixed_visual_screen_passed':True,'candidate':doc['candidate']}))
PY
echo "$(date -Iseconds) mixed_visual_screen_gate_passed"

nohup env TRACK2_ROOT="$ROOT" "$ONE_UPDATE" \
  > "$RESOURCE_RUN/audit/v166_one_update_launcher.log" 2>&1 &
run_pid=$!
printf '%s\n' "$run_pid" > "$RESOURCE_RUN/audit/v166_one_update.pid"
echo "$(date -Iseconds) v166_one_update_started pid=$run_pid"
