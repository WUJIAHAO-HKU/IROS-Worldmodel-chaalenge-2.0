#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$BASE/artifacts/strict_track2_official_20260810"
NAME='v263_v261_fullright_chunk8_sft2048_joint4_grip4_lr1e6_seed1463_20260819'
REG="$OFF/run_registry/$NAME"
V261_CKPT="$OFF/runs/v261_v169step5_fullright_chunk8_sft512_grip8_inactive1_seed1462_20260819/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt"

test ! -e "$OFF/runs/$NAME"
test ! -e "$REG"
test -s "$V261_CKPT"

screen -dmS v263_fullright_joint_routing_sft bash -lc \
  "TRACK2_V259_NAME='$NAME' \
TRACK2_V259_VARIANT='v263_fullright_chunk8_sft2048_j4g4_seed1463' \
TRACK2_V259_BASE_CKPT='$V261_CKPT' \
TRACK2_V259_DATA='/root/autodl-tmp/iros_v15_rl_probability_audit_v2/artifacts/lerobot/adjust_bottle_train40_arm1' \
TRACK2_V259_DATA_AUDIT='$OFF/diagnostics/full_right_sft_dataset_audit_20260819.json' \
TRACK2_V259_CONVERSION='/root/autodl-tmp/iros_v15_rl_probability_audit_v2/artifacts/lerobot/adjust_bottle_train40_arm1/conversion_summary.json' \
TRACK2_V259_EVIDENCE_AUDIT='$OFF/diagnostics/v262_v261_right_routing_failure_20260819.json' \
TRACK2_V259_PREPARE_SCRIPT='prepare_v263_fullright_joint_routing_sft.py' \
TRACK2_V259_ACTOR_SEED=1463 TRACK2_V259_EXTRA_UPDATES=2048 \
TRACK2_V259_ACTOR_LR=1e-6 TRACK2_V259_KL_BETA=0.2 \
TRACK2_V259_SFT_LOSS_WEIGHT=0.3 TRACK2_V259_SFT_EXTRA_LOSS_WEIGHT=1.0 \
TRACK2_V259_ACTIVE_JOINT_WEIGHT=4.0 TRACK2_V259_ACTIVE_GRIPPER_WEIGHT=4.0 \
TRACK2_V259_INACTIVE_KEEP_WEIGHT=1.0 \
bash '$BASE/pipeline/scripts/launch_v259_chunk8_right_sft.sh' >> '$REG.console.log' 2>&1"
sleep 2
screen -ls | grep -q v263_fullright_joint_routing_sft
echo V263_FULLRIGHT_JOINT_ROUTING_SFT_STARTED
