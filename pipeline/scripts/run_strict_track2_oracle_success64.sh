#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-/root/autodl-tmp/IROS_WAM_2.0 challenge}
ROBOTWIN_ROOT="$ROOT/artifacts/strict_track2_official_20260810/official_deps/RoboTwin_RLinf_support"
JOINT="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
CONFIG_NAME=strict_track2_oracle_success64
SOURCE_CONFIG="$ROOT/pipeline/config/$CONFIG_NAME.yml"
RUNTIME_CONFIG="$ROBOTWIN_ROOT/task_config/$CONFIG_NAME.yml"
PREREG="$ROOT/pipeline/config/strict_track2_oracle_success64_preregistration.json"
OUTPUT="$JOINT/robotwin_oracle_success64/adjust_bottle/$CONFIG_NAME"
CONTROL="$JOINT/robotwin_oracle_success64_control"
PYTHON_BIN=/root/autodl-tmp/conda_envs/rlinf_track2/bin/python

if pgrep -af '[e]val_embodied_agent.py|[t]rain_embodied_agent.py|[e]valuate_strict_track2_v15' >/dev/null; then
  printf 'Refusing to contend with active Track 2 policy or parent evaluation\n' >&2
  exit 3
fi
test -s "$SOURCE_CONFIG"
test -s "$PREREG"
test ! -e "$OUTPUT"
mkdir -p "$CONTROL" "$JOINT/audit"
cp "$SOURCE_CONFIG" "$RUNTIME_CONFIG"
cp "$SOURCE_CONFIG" "$CONTROL/"
cp "$PREREG" "$CONTROL/"
{
  printf 'format=strict-track2-public-robotwin-oracle-success64-v1\n'
  printf 'source=public RoboTwin simulator, mplib expert, and public assets\n'
  printf 'world_model_training_only=true\n'
  printf 'policy_action_injection=false\n'
  printf 'mpc=false\n'
  printf 'robotwin_commit=%s\n' "$(git -C "$ROBOTWIN_ROOT" rev-parse HEAD)"
  printf 'config_sha256=%s\n' "$(sha256sum "$SOURCE_CONFIG" | cut -d' ' -f1)"
  printf 'preregistration_sha256=%s\n' "$(sha256sum "$PREREG" | cut -d' ' -f1)"
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
} >"$CONTROL/provenance.txt"

cd "$ROBOTWIN_ROOT"
PYTHONHASHSEED=0 \
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
XDG_RUNTIME_DIR=/tmp/xdg-track2-oracle-success64 \
"$PYTHON_BIN" script/collect_data.py adjust_bottle "$CONFIG_NAME" \
  >"$CONTROL/collection.log" 2>&1

DATA="$OUTPUT/data"
test -s "$OUTPUT/seed.txt"
count=$(find "$DATA" -maxdepth 1 -type f -name 'episode*.hdf5' | wc -l)
if [[ "$count" -ne 64 ]]; then
  printf 'Expected 64 oracle episodes, found %s\n' "$count" >&2
  exit 5
fi
{
  printf 'completed_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'hdf5_count=%s\n' "$count"
  printf 'seed_sha256=%s\n' "$(sha256sum "$OUTPUT/seed.txt" | cut -d' ' -f1)"
  printf 'data_merkle_sha256=%s\n' "$({ find "$DATA" -maxdepth 1 -type f -name 'episode*.hdf5' -print0 | sort -z | xargs -0 sha256sum; } | sha256sum | cut -d' ' -f1)"
} >>"$CONTROL/provenance.txt"
printf 'ORACLE_SUCCESS64_COLLECTION_COMPLETE output=%s\n' "$OUTPUT"
