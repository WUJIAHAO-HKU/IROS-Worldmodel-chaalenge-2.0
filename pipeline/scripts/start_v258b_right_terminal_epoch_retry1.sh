#!/usr/bin/env bash
set -euo pipefail
ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$ROOT/artifacts/strict_track2_official_20260810"
NAME='v258b_v169step5_rightterminal_sft1epoch_h8_step1_extra638_lr5e6_seed1460_retry1_20260819'
VARIANT='v258b_rightterminal_epoch1_seed1460_retry1'
REG="$OFF/run_registry/$NAME"
RETRY_OF="$OFF/run_registry/v258_v169step5_rightterminal_sft1epoch_h8_step1_extra638_lr5e6_seed1460_20260819/preregistration.json"
test ! -e "$REG"
test -f "$RETRY_OF"
screen -dmS v258b_right_terminal_epoch_retry1 bash -lc \
  "TRACK2_V258_NAME='$NAME' TRACK2_V258_VARIANT='$VARIANT' TRACK2_V258_RETRY_OF='$RETRY_OF' bash '$ROOT/pipeline/scripts/launch_v258_right_terminal_epoch.sh' >> '$OFF/run_registry/$NAME.console.log' 2>&1"
sleep 2
screen -ls | grep -q v258b_right_terminal_epoch_retry1
echo V258B_RIGHT_TERMINAL_EPOCH_RETRY1_STARTED
