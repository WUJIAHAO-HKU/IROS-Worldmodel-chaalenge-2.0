#!/usr/bin/env bash
set -euo pipefail
BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$BASE/artifacts/strict_track2_official_20260810"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
SUMMARY='/root/autodl-tmp/iros_v15_rl_probability_audit_v2/artifacts/lerobot/adjust_bottle_train40_arm1/conversion_summary.json'
OUTPUT='/root/autodl-tmp/iros_v15_rl_probability_audit_v2/artifacts/lerobot/adjust_bottle_train40_right_terminal_v1'
AUDIT="$OFF/run_registry/v225_right_terminal_dataset_20260818"
test ! -e "$OUTPUT"
test ! -e "$AUDIT"
mkdir -p "$AUDIT"
"$PY" "$BASE/pipeline/scripts/build_right_terminal_lerobot_dataset.py" "$SUMMARY" "$OUTPUT" \
  > "$AUDIT/build.log" 2>&1
find "$OUTPUT" -type f -print0 | sort -z | xargs -0 sha256sum > "$AUDIT/dataset_sha256.txt"
"$PY" - "$OUTPUT" "$AUDIT/dataset_audit.json" <<'PY'
import json, pathlib, sys
root, out = map(pathlib.Path, sys.argv[1:])
summary = json.loads((root / 'conversion_summary.json').read_text())
info = json.loads((root / 'meta/info.json').read_text())
assert summary['episodes'] == info['total_episodes'] == 15
assert summary['frames'] == info['total_frames']
assert all(item['selected_postclose_fraction'] > 0.89 for item in summary['records'])
report = {
    'accepted': True,
    'episodes': info['total_episodes'],
    'frames': info['total_frames'],
    'tasks': info['total_tasks'],
    'min_selected_postclose_fraction': min(item['selected_postclose_fraction'] for item in summary['records']),
    'source_public_only': True,
    'reserved_final128_access': False,
    'real_competition_submission': False,
}
out.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
PY
touch "$AUDIT/V225_RIGHT_TERMINAL_DATASET_ACCEPTED"
echo V225_RIGHT_TERMINAL_DATASET_ACCEPTED
