#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$BASE/artifacts/strict_track2_official_20260810"
NAME='v268_v265_mixedarm_structured_chunk8_sft2334_b4_lr1e6_seed1468_20260819'
REG="$OFF/run_registry/$NAME"
V265="$OFF/runs/v265_v169step5_mirrorbalanced_chunk8_sft2334_b4_lr2e6_seed1465_20260819"
V265_CKPT="$V265/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt"

test ! -e "$OFF/runs/$NAME"
test ! -e "$REG"
test -s "$V265_CKPT"
screen -dmS v268_mixed_arm_structured_sft bash -lc \
  "TRACK2_V259_NAME='$NAME' \
TRACK2_V259_VARIANT='v268_mixedarm_structured_epoch1_seed1468' \
TRACK2_V259_BASE_CKPT='$V265_CKPT' \
TRACK2_V259_DATA='/root/autodl-tmp/iros_v15_rl_probability_audit_v2/artifacts/lerobot/adjust_bottle_train40_mirror_balanced_v1' \
TRACK2_V259_DATA_AUDIT='$OFF/diagnostics/mirror_balanced_sft_dataset_audit_20260819.json' \
TRACK2_V259_CONVERSION='/root/autodl-tmp/iros_v15_rl_probability_audit_v2/artifacts/lerobot/adjust_bottle_train40_mirror_balanced_v1/conversion_summary.json' \
TRACK2_V259_EVIDENCE_AUDIT='$OFF/diagnostics/mixed_arm_action_inference_zero_baseline_audit_20260819.json' \
TRACK2_V259_PREPARE_SCRIPT='prepare_v268_mixed_arm_structured_sft.py' \
TRACK2_V259_ACTOR_SEED=1468 TRACK2_V259_EXTRA_UPDATES=2334 \
TRACK2_V259_ACTOR_LR=1e-6 TRACK2_V259_KL_BETA=0.2 \
TRACK2_V259_SFT_BATCH_SIZE=4 TRACK2_V259_USE_STRUCTURED_SFT_ACTION_LOSS=false \
TRACK2_V259_USE_MIXED_ARM_STRUCTURED_SFT_ACTION_LOSS=true \
TRACK2_V259_ACTIVE_JOINT_WEIGHT=1.0 TRACK2_V259_ACTIVE_GRIPPER_WEIGHT=3.0 \
TRACK2_V259_INACTIVE_KEEP_WEIGHT=2.0 TRACK2_V259_GRIPPER_ACTIVITY_WEIGHT=0.25 \
TRACK2_V259_SFT_LOSS_WEIGHT=0.3 TRACK2_V259_SFT_EXTRA_LOSS_WEIGHT=1.0 \
bash '$BASE/pipeline/scripts/launch_v259_chunk8_right_sft.sh' >> '$REG.console.log' 2>&1"
sleep 2
screen -ls | grep -q v268_mixed_arm_structured_sft
echo V268_MIXED_ARM_STRUCTURED_SFT_STARTED
