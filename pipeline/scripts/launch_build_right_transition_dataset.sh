#!/usr/bin/env bash
set -euo pipefail
BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$BASE/artifacts/strict_track2_official_20260810"
REG="$OFF/run_registry/v231b_right_transition_dataset_20260818"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
SUMMARY='/root/autodl-tmp/iros_v15_rl_probability_audit_v2/artifacts/lerobot/adjust_bottle_train40_arm1/conversion_summary.json'
OUT='/root/autodl-tmp/iros_v15_rl_probability_audit_v2/artifacts/lerobot/adjust_bottle_train40_right_transition_v1'
test ! -e "$REG"
test ! -e "$OUT"
mkdir -p "$REG"
"$PY" '/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/build_right_transition_lerobot_dataset.py' "$SUMMARY" "$OUT" > "$REG/build.log" 2>&1
"$PY" - "$OUT/conversion_summary.json" "$REG/dataset_audit.json" <<'PY'
import json, pathlib, sys
source, output = map(pathlib.Path, sys.argv[1:])
data = json.loads(source.read_text())
report = {
    "accepted": data["episodes"] == 15 and data["frames"] == 795 and all(r["selected_frames"] == 53 for r in data["records"]),
    "episodes": data["episodes"], "frames": data["frames"],
    "selection": data["selection"], "source_public_only": True,
    "reserved_final128_access": False, "real_competition_submission": False,
}
output.write_text(json.dumps(report, indent=2) + "\n")
assert report["accepted"]
PY
find "$OUT" -type f -print0 | sort -z | xargs -0 sha256sum > "$REG/dataset_sha256.txt"
touch "$REG/V231B_RIGHT_TRANSITION_DATASET_ACCEPTED"
echo V231B_RIGHT_TRANSITION_DATASET_ACCEPTED
