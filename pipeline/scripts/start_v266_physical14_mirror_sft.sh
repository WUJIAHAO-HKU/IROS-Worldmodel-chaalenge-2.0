#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$BASE/artifacts/strict_track2_official_20260810"
NAME='v266_v265_physical14_mirror_chunk8_sft2334_b4_lr1e6_seed1466_20260819'
REG="$OFF/run_registry/$NAME"
V265="$OFF/runs/v265_v169step5_mirrorbalanced_chunk8_sft2334_b4_lr2e6_seed1465_20260819"
V265_CKPT="$V265/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt"

test ! -e "$OFF/runs/$NAME"
test ! -e "$REG"
test -s "$V265_CKPT"
screen -dmS v266_physical14_mirror_sft bash -lc \
  "TRACK2_V259_NAME='$NAME' \
TRACK2_V259_VARIANT='v266_physical14_mirror_epoch1_seed1466' \
TRACK2_V259_BASE_CKPT='$V265_CKPT' \
TRACK2_V259_DATA='/root/autodl-tmp/iros_v15_rl_probability_audit_v2/artifacts/lerobot/adjust_bottle_train40_mirror_balanced_v1' \
TRACK2_V259_DATA_AUDIT='$OFF/diagnostics/mirror_balanced_sft_dataset_audit_20260819.json' \
TRACK2_V259_CONVERSION='/root/autodl-tmp/iros_v15_rl_probability_audit_v2/artifacts/lerobot/adjust_bottle_train40_mirror_balanced_v1/conversion_summary.json' \
TRACK2_V259_EVIDENCE_AUDIT='$V265/audit/public_right_r0_acceptance.json' \
TRACK2_V259_PREPARE_SCRIPT='prepare_v266_physical14_mirror_sft.py' \
TRACK2_V259_ACTOR_SEED=1466 TRACK2_V259_EXTRA_UPDATES=2334 \
TRACK2_V259_ACTOR_LR=1e-6 TRACK2_V259_KL_BETA=0.2 \
TRACK2_V259_SFT_BATCH_SIZE=4 TRACK2_V259_USE_STRUCTURED_SFT_ACTION_LOSS=false \
TRACK2_V259_SFT_PHYSICAL_ACTION_DIM=14 \
TRACK2_V259_SFT_LOSS_WEIGHT=0.3 TRACK2_V259_SFT_EXTRA_LOSS_WEIGHT=1.0 \
bash '$BASE/pipeline/scripts/launch_v259_chunk8_right_sft.sh' >> '$REG.console.log' 2>&1"
sleep 2
screen -ls | grep -q v266_physical14_mirror_sft
echo V266_PHYSICAL14_MIRROR_SFT_STARTED
