#!/usr/bin/env bash
set -euo pipefail

cd "/root/autodl-tmp/IROS_WAM_2.0 challenge"
run="artifacts/strict_track2_joint_augmentation_20260810/v339_high_specificity_recursive_ood_gate_seed1508_20260822"
source_run="artifacts/strict_track2_joint_augmentation_20260810/v337_public_recursive_ood_gate_seed1507_20260822"
mkdir -p "$run"
export PYTHONPATH="pipeline:pipeline/scripts"
/root/miniconda3/envs/go1/bin/python pipeline/scripts/prepare_v339_high_specificity_recursive_ood_gate.py
exec taskset -c 0-7 /root/miniconda3/envs/go1/bin/python \
  pipeline/scripts/audit_v339_recalibrated_holdout.py \
  --gate "$run/recursive_ood_gate.npz" \
  --calibration-report "$run/calibration_report.json" \
  --source-holdout-report "$source_run/holdout_report.json" \
  --output "$run/holdout_report.json"
