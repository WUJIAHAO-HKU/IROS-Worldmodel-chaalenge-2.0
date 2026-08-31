#!/usr/bin/env bash
set -euo pipefail

cd '/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF='artifacts/strict_track2_official_20260810'
NAME='v318_v317_authorized_official_h200_r4_step10_lr2e5_beta001_seed1471_20260822'
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"
EXP='wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05'
PREFIX="$RUN/audit/v318_training_prefix_step5.json"
CKPT="$RUN/$EXP/checkpoints/global_step_6/actor/model_state_dict/full_weights.pt"
OPT="$RUN/$EXP/checkpoints/global_step_6/actor/optimizer_recovery.pt"
REPORT="$RUN/audit/v318_global_step6_continuation_gate.json"
PREREG="$REG/global_step6_continuation_gate_preregistration.json"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'

for _ in $(seq 1 360); do
  if [[ -s "$PREFIX" && -s "$CKPT" && -s "$OPT" ]]; then
    old_model=$(stat -c '%s:%Y' "$CKPT"); old_opt=$(stat -c '%s:%Y' "$OPT")
    sleep 30
    new_model=$(stat -c '%s:%Y' "$CKPT"); new_opt=$(stat -c '%s:%Y' "$OPT")
    if [[ "$old_model" == "$new_model" && "$old_opt" == "$new_opt" ]]; then
      set +e
      "$PY" pipeline/scripts/audit_v318_step6_stop_gate.py --preregistration "$PREREG" \
        --prefix-audit "$PREFIX" --checkpoint "$CKPT" --optimizer "$OPT" --output "$REPORT" \
        >"$RUN/audit/v318_global_step6_continuation_gate.log" 2>&1
      rc=$?
      set -e
      if (( rc == 0 )); then
        touch "$REG/V318_GLOBAL_STEP6_GATE_PASSED"
        echo V318_GLOBAL_STEP6_GATE_PASSED
        exit 0
      fi
      touch "$REG/V318_GLOBAL_STEP6_GATE_FAILED"
      screen -S v318_resume_from_step3 -X quit >/dev/null 2>&1 || true
      echo V318_GLOBAL_STEP6_GATE_FAILED_AND_TRAINING_STOPPED
      exit "$rc"
    fi
  fi
  if ! screen -ls 2>/dev/null | grep -q '[.]v318_resume_from_step3'; then exit 3; fi
  sleep 30
done
exit 5
