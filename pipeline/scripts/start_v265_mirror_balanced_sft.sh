#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$BASE/artifacts/strict_track2_official_20260810"
NAME='v265_v169step5_mirrorbalanced_chunk8_sft2334_b4_lr2e6_seed1465_20260819'
REG="$OFF/run_registry/$NAME"

test ! -e "$OFF/runs/$NAME"
test ! -e "$REG"
screen -dmS v265_mirror_balanced_sft bash -lc \
  "TRACK2_V259_NAME='$NAME' \
TRACK2_V259_VARIANT='v265_mirrorbalanced_chunk8_epoch1_seed1465' \
TRACK2_V259_DATA='/root/autodl-tmp/iros_v15_rl_probability_audit_v2/artifacts/lerobot/adjust_bottle_train40_mirror_balanced_v1' \
TRACK2_V259_DATA_AUDIT='$OFF/diagnostics/mirror_balanced_sft_dataset_audit_20260819.json' \
TRACK2_V259_CONVERSION='/root/autodl-tmp/iros_v15_rl_probability_audit_v2/artifacts/lerobot/adjust_bottle_train40_mirror_balanced_v1/conversion_summary.json' \
TRACK2_V259_EVIDENCE_AUDIT='$OFF/diagnostics/v262_v261_right_routing_failure_20260819.json' \
TRACK2_V259_PREPARE_SCRIPT='prepare_v265_mirror_balanced_sft.py' \
TRACK2_V259_ACTOR_SEED=1465 TRACK2_V259_EXTRA_UPDATES=2334 \
TRACK2_V259_ACTOR_LR=2e-6 TRACK2_V259_KL_BETA=0.1 \
TRACK2_V259_SFT_BATCH_SIZE=4 TRACK2_V259_USE_STRUCTURED_SFT_ACTION_LOSS=false \
TRACK2_V259_SFT_LOSS_WEIGHT=0.3 TRACK2_V259_SFT_EXTRA_LOSS_WEIGHT=1.0 \
bash '$BASE/pipeline/scripts/launch_v259_chunk8_right_sft.sh' >> '$REG.console.log' 2>&1"
sleep 2
screen -ls | grep -q v265_mirror_balanced_sft
echo V265_MIRROR_BALANCED_SFT_STARTED
