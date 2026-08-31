#!/usr/bin/env bash
set -euo pipefail

PROJECT="/root/autodl-tmp/IROS_WAM_2.0 challenge"
PYTHON="/root/autodl-tmp/conda_envs/rlinf_track2/bin/python"
OUTPUT="/root/autodl-tmp/iros_v15_rl_probability_audit_v2/artifacts/lerobot/adjust_bottle_train40_armconsistent_transition_balanced_v2"
LOG="/root/autodl-tmp/v370_dataset_build.log"
EXIT_FILE="/root/autodl-tmp/v370_dataset_build.exit"

cd "$PROJECT"
if [[ -e "$OUTPUT" ]]; then
  echo "refusing to overwrite $OUTPUT" >&2
  exit 9
fi
rm -f "$LOG" "$EXIT_FILE"
set +e
taskset -c 0-7 "$PYTHON" pipeline/scripts/build_v370_armconsistent_transition_balanced_lerobot.py \
  --source artifacts/datasets/aloha-agilex_clean_50 \
  --split artifacts/strict_track2_joint_augmentation_20260810/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json \
  --output "$OUTPUT" \
  --right-repeats 4 >"$LOG" 2>&1
status=$?
set -e
printf '%s\n' "$status" >"$EXIT_FILE"
exit "$status"
