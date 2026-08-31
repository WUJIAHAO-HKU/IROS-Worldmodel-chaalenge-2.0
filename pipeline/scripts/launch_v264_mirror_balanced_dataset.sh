#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$BASE/artifacts/strict_track2_official_20260810"
P="$BASE/pipeline/scripts"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
NAME='v264_train40_mirror_balanced_dataset_20260819'
REG="$OFF/run_registry/$NAME"
SOURCE="$BASE/artifacts/datasets/aloha-agilex_clean_50"
SPLIT="$BASE/artifacts/splits/adjust_bottle_50episodes_full.json"
RESET="$BASE/artifacts/rlinf_public_reset_adjust_bottle"
MIRROR_AUDIT="$OFF/diagnostics/train40_mirror_augmentation_audit_20260819.json"
DATA='/root/autodl-tmp/iros_v15_rl_probability_audit_v2/artifacts/lerobot/adjust_bottle_train40_mirror_balanced_v1'
DATA_AUDIT="$OFF/diagnostics/mirror_balanced_sft_dataset_audit_20260819.json"

test ! -e "$REG"
test ! -e "$DATA"
test ! -e "$DATA_AUDIT"
"$PY" "$P/prepare_v264_mirror_balanced_dataset.py" "$REG/preregistration.json" \
  "$MIRROR_AUDIT" "$P/build_train40_mirror_balanced_lerobot.py" \
  "$P/audit_mirror_balanced_lerobot_dataset.py" "$DATA" > /tmp/v264_prepare.log
mv /tmp/v264_prepare.log "$REG/prepare.log"

"$PY" "$P/build_train40_mirror_balanced_lerobot.py" \
  --source "$SOURCE" --split "$SPLIT" --reset-dir "$RESET" \
  --audit "$MIRROR_AUDIT" --output "$DATA" >"$REG/build.log" 2>&1
"$PY" "$P/audit_mirror_balanced_lerobot_dataset.py" \
  --dataset "$DATA" --output "$DATA_AUDIT" >"$REG/audit.log" 2>&1
touch "$REG/V264_MIRROR_BALANCED_DATASET_ACCEPTED"
echo V264_MIRROR_BALANCED_DATASET_ACCEPTED
