#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$BASE/artifacts/strict_track2_official_20260810"
NAME='v372r_v371_deterministic_checkpoint_recovery_seed1535_20260823'
SCREEN_NAME='v372r_checkpoint_recovery'
REG="$OFF/run_registry/$NAME"
BASE_CKPT="$OFF/runs/v169_parent_step5_seed1243_20260813/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt"
DATA='/root/autodl-tmp/iros_v15_rl_probability_audit_v2/artifacts/lerobot/adjust_bottle_train40_armconsistent_transition_balanced_v2'
DATA_AUDIT="$BASE/artifacts/strict_track2_joint_augmentation_20260810/run_registry/v370b_armconsistent_transition_balanced_dataset_seed1534_20260822/dataset_audit.json"
EVIDENCE="$OFF/real_robotwin_eval/v169_four_step_retry2_summary.json"

if [[ "${1:-}" != '--run' ]]; then
  test ! -e "$OFF/runs/$NAME"
  test ! -e "$REG"
  test -s "$BASE_CKPT"
  test -f "$DATA/conversion_summary.json"
  test -f "$DATA_AUDIT"
  avail_bytes=$(df -B1 --output=avail /root/autodl-tmp | tail -1)
  if (( avail_bytes < 14000000000 )); then
    echo "v372r requires at least 14 GB free before launch; available=$avail_bytes" >&2
    exit 12
  fi
  screen -dmS "$SCREEN_NAME" "$0" --run
  sleep 3
  screen -ls | grep -q "$SCREEN_NAME"
  echo V372R_DETERMINISTIC_CHECKPOINT_RECOVERY_STARTED
  exit 0
fi

export TRACK2_V259_NAME="$NAME"
export TRACK2_V259_VARIANT='v372r_armconsistent_transitionbalanced_epoch1_seed1535'
export TRACK2_V259_BASE_CKPT="$BASE_CKPT"
export TRACK2_V259_DATA="$DATA"
export TRACK2_V259_DATA_AUDIT="$DATA_AUDIT"
export TRACK2_V259_CONVERSION="$DATA/conversion_summary.json"
export TRACK2_V259_EVIDENCE_AUDIT="$EVIDENCE"
export TRACK2_V259_PREPARE_SCRIPT='prepare_v371_armconsistent_transitionbalanced_sft.py'
export TRACK2_V259_ACTOR_SEED=1535
export TRACK2_V259_EXTRA_UPDATES=1697
export TRACK2_V259_ACTOR_LR=1e-6
export TRACK2_V259_KL_BETA=0.2
export TRACK2_V259_SFT_BATCH_SIZE=4
export TRACK2_V259_USE_STRUCTURED_SFT_ACTION_LOSS=false
export TRACK2_V259_USE_MIXED_ARM_STRUCTURED_SFT_ACTION_LOSS=true
export TRACK2_V259_ACTIVE_JOINT_WEIGHT=1.0
export TRACK2_V259_ACTIVE_GRIPPER_WEIGHT=3.0
export TRACK2_V259_INACTIVE_KEEP_WEIGHT=2.0
export TRACK2_V259_GRIPPER_ACTIVITY_WEIGHT=0.25
export TRACK2_V259_SFT_LOSS_WEIGHT=0.3
export TRACK2_V259_SFT_EXTRA_LOSS_WEIGHT=1.0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

exec taskset -c 0-21 bash "$BASE/pipeline/scripts/launch_v259_chunk8_right_sft.sh"
