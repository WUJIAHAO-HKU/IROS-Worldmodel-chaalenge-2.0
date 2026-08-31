#!/usr/bin/env bash
set -euo pipefail
ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$ROOT/artifacts/strict_track2_official_20260810"
NAME='v259_v169step5_rightterminal_chunk8_sft128_h8_step1_lr5e6_seed1461_20260819'
REG="$OFF/run_registry/$NAME"
test ! -e "$REG"
screen -dmS v259_chunk8_right_sft bash -lc \
  "bash '$ROOT/pipeline/scripts/launch_v259_chunk8_right_sft.sh' >> '$OFF/run_registry/$NAME.console.log' 2>&1"
sleep 2
screen -ls | grep -q v259_chunk8_right_sft
echo V259_CHUNK8_RIGHT_SFT_STARTED
