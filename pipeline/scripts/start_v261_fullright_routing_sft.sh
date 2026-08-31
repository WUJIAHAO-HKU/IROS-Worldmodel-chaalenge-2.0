#!/usr/bin/env bash
set -euo pipefail
ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$ROOT/artifacts/strict_track2_official_20260810"
NAME='v261_v169step5_fullright_chunk8_sft512_grip8_inactive1_seed1462_20260819'
VARIANT='v261_fullright_chunk8_sft512_grip8_seed1462'
REG="$OFF/run_registry/$NAME"
ROUTING="$OFF/diagnostics/v260_v259_right_routing_failure_20260819.json"
test ! -e "$REG"
test -f "$ROUTING"
screen -dmS v261_fullright_routing_sft bash -lc \
  "TRACK2_V259_NAME='$NAME' \
TRACK2_V259_VARIANT='$VARIANT' \
TRACK2_V259_DATA='/root/autodl-tmp/iros_v15_rl_probability_audit_v2/artifacts/lerobot/adjust_bottle_train40_arm1' \
TRACK2_V259_DATA_AUDIT='$OFF/diagnostics/full_right_sft_dataset_audit_20260819.json' \
TRACK2_V259_CONVERSION='/root/autodl-tmp/iros_v15_rl_probability_audit_v2/artifacts/lerobot/adjust_bottle_train40_arm1/conversion_summary.json' \
TRACK2_V259_EVIDENCE_AUDIT='$ROUTING' \
TRACK2_V259_PREPARE_SCRIPT='prepare_v261_fullright_routing_sft.py' \
TRACK2_V259_ACTOR_SEED=1462 TRACK2_V259_EXTRA_UPDATES=512 \
TRACK2_V259_ACTIVE_JOINT_WEIGHT=1.0 TRACK2_V259_ACTIVE_GRIPPER_WEIGHT=8.0 \
TRACK2_V259_INACTIVE_KEEP_WEIGHT=1.0 \
bash '$ROOT/pipeline/scripts/launch_v259_chunk8_right_sft.sh' \
>> '$OFF/run_registry/$NAME.console.log' 2>&1"
sleep 2
screen -ls | grep -q v261_fullright_routing_sft
echo V261_FULLRIGHT_ROUTING_SFT_STARTED
