#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$BASE/artifacts/strict_track2_official_20260810"
NAME='v264_train40_mirror_balanced_dataset_20260819'
test ! -e "$OFF/run_registry/$NAME"
screen -dmS v264_mirror_balanced_dataset bash -lc \
  "bash '$BASE/pipeline/scripts/launch_v264_mirror_balanced_dataset.sh' >> '$OFF/run_registry/$NAME.console.log' 2>&1"
sleep 2
screen -ls | grep -q v264_mirror_balanced_dataset
echo V264_MIRROR_BALANCED_DATASET_STARTED
