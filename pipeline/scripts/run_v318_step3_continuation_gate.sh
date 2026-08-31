#!/usr/bin/env bash
set -euo pipefail

cd '/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF='/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_official_20260810'
NAME='v318_v317_authorized_official_h200_r4_step10_lr2e5_beta001_seed1471_20260822'
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"
EXP='wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05'
CKPT="$RUN/$EXP/checkpoints/global_step_3/actor/model_state_dict/full_weights.pt"
OPT="$RUN/$EXP/checkpoints/global_step_3/actor/optimizer_recovery.pt"
PREFIX="$RUN/audit/v318_training_prefix_step2.json"
PREREG="$REG/global_step3_continuation_gate_preregistration_v3.json"
OFFLINE="$RUN/audit/v318_global_step3_offline_policy_trainfit.json"
REPORT="$RUN/audit/v318_global_step3_continuation_gate.json"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
EVAL_PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
DATA='/root/autodl-tmp/iros_v15_rl_probability_audit_v2/artifacts/lerobot/adjust_bottle_train40_mirror_balanced_v1'
OFFICIAL='/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/official_resources/pi05_adjust_bottle'
REFERENCE="$OFF/immutable_reference_policy_state/pi05_official_rank0.pt"
RLINF='/root/autodl-tmp/IROS_WAM_2.0 challenge/third_party/WorldArena-2.0/RL_env_benchmark'
OPENPI='/root/autodl-tmp/IROS_WAM_2.0 challenge/third_party/openpi-rlinf-full'
DIFFSYNTH='/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_official_20260810/official_deps/diffsynth_2a2e05f'

test -s "$REG/V318_PAUSED_AT_GLOBAL_STEP3"
for path in "$CKPT" "$OPT" "$PREFIX" "$PREREG" "$REFERENCE" "$DATA/conversion_summary.json"; do
  test -s "$path"
done
test ! -e "$OFFLINE"
test ! -e "$REPORT"

"$PY" - "$PREREG" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=json.load(open(sys.argv[1]))
for key in ("offline_policy_auditor", "continuation_auditor"):
    item=p["inputs"][key]; path=Path(item["path"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], key
PY

PYTHONPATH="$DIFFSYNTH:$OPENPI/packages/openpi-client/src:$OPENPI/src:$RLINF" \
taskset -c 0-21 "$EVAL_PY" pipeline/scripts/audit_v318_offline_policy_trainfit.py \
  --official "$OFFICIAL" --dataset "$DATA" \
  --checkpoint "official=$REFERENCE" --checkpoint "v318_step3=$CKPT" \
  --records-per-arm 4 --batch-size 4 --output "$OFFLINE" \
  >"$RUN/audit/v318_global_step3_offline_policy_trainfit.log" 2>&1

set +e
"$PY" pipeline/scripts/audit_v318_step3_stop_gate.py \
  --preregistration "$PREREG" --prefix-audit "$PREFIX" --offline-audit "$OFFLINE" \
  --checkpoint "$CKPT" --optimizer "$OPT" --output "$REPORT" \
  >"$RUN/audit/v318_global_step3_continuation_gate.log" 2>&1
rc=$?
set -e

if (( rc == 0 )); then
  touch "$REG/V318_GLOBAL_STEP3_GATE_PASSED"
  screen -S v318_resume_from_step3 -X quit >/dev/null 2>&1 || true
  screen -L -Logfile "$REG/step3_resume.screen.log" -dmS v318_resume_from_step3 \
    bash pipeline/scripts/launch_v318_resume_from_step3.sh
  echo V318_GLOBAL_STEP3_GATE_PASSED_AND_RESUMED
  exit 0
fi

touch "$REG/V318_GLOBAL_STEP3_GATE_FAILED"
screen -S v318_v317_official_rl -X quit >/dev/null 2>&1 || true
bash pipeline/scripts/restart_v317_services.sh stop >/dev/null 2>&1 || true
TRACK2_CPUSET=0-15 bash pipeline/scripts/restart_v271_v274_services.sh start \
  >"$REG/restart_v271_after_step3_rejection.log" 2>&1 || true
echo V318_GLOBAL_STEP3_GATE_FAILED_AND_REJECTED
exit "$rc"
