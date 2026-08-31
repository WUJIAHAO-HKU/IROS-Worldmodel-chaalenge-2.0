#!/usr/bin/env bash
set -euo pipefail

cd '/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF='artifacts/strict_track2_official_20260810'
NAME='v318_v317_authorized_official_h200_r4_step10_lr2e5_beta001_seed1471_20260822'
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"
CKPT="$RUN/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_10/actor/model_state_dict/full_weights.pt"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'

for _ in $(seq 1 1440); do
  if [[ -f "$RUN/audit/V318_TRAINING_ACCEPTED" && -s "$CKPT" ]]; then
    "$PY" -c 'import sys,zipfile; raise SystemExit(0 if zipfile.is_zipfile(sys.argv[1]) else 1)' "$CKPT"
    test -s "$RUN/audit/p3_training_acceptance.json"
    screen -S v318_posttraining_pipeline -X quit >/dev/null 2>&1 || true
    screen -L -Logfile "$REG/posttraining.screen.log" -dmS v318_posttraining_pipeline \
      bash pipeline/scripts/run_v318_posttraining_pipeline.sh
    touch "$REG/V318_POSTTRAINING_PIPELINE_LAUNCHED"
    echo V318_POSTTRAINING_PIPELINE_LAUNCHED
    exit 0
  fi
  if [[ -f "$REG/V318_GLOBAL_STEP3_GATE_FAILED" ]]; then exit 3; fi
  if [[ -f "$REG/V318_PREFIX_AUDIT_FAILED_STEP_3" || -f "$REG/V318_PREFIX_AUDIT_FAILED_STEP_4" || -f "$REG/V318_PREFIX_AUDIT_FAILED_STEP_5" || -f "$REG/V318_PREFIX_AUDIT_FAILED_STEP_6" || -f "$REG/V318_PREFIX_AUDIT_FAILED_STEP_7" || -f "$REG/V318_PREFIX_AUDIT_FAILED_STEP_8" || -f "$REG/V318_PREFIX_AUDIT_FAILED_STEP_9" ]]; then
    echo V318_PREFIX_FAILED
    exit 4
  fi
  if ! screen -ls 2>/dev/null | grep -q '[.]v318_resume_from_step3'; then
    if [[ ! -f "$RUN/audit/V318_TRAINING_ACCEPTED" ]]; then
      echo V318_RESUME_STOPPED_WITHOUT_ACCEPTANCE
      exit 5
    fi
  fi
  sleep 60
done

exit 6
