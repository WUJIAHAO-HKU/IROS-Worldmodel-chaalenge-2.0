#!/usr/bin/env bash
set -euo pipefail
ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
NAME='v258_v169step5_rightterminal_sft1epoch_h8_step1_extra638_lr5e6_seed1460_20260819'
REG="$ROOT/artifacts/strict_track2_official_20260810/run_registry/$NAME"
test ! -e "$REG"
screen -dmS v258_right_terminal_epoch bash -lc \
  "bash '$ROOT/pipeline/scripts/launch_v258_right_terminal_epoch.sh' >> '$ROOT/artifacts/strict_track2_official_20260810/run_registry/$NAME.console.log' 2>&1"
sleep 2
screen -ls | grep -q v258_right_terminal_epoch
echo V258_RIGHT_TERMINAL_EPOCH_STARTED
