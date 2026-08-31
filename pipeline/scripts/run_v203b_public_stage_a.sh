#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$BASE/artifacts/strict_track2_official_20260810"
NAME='v203b_v202_conservativekl_h200_r2_step8_seed1403_20260818'
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
CHECKPOINT="$RUN/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_8/actor/model_state_dict/full_weights.pt"
TRAIN_AUDIT="$RUN/audit/p3_training_acceptance.json"
TRAIN_PREREG="$REG/preregistration.json"
DEV32="$OFF/real_robotwin_eval/public_unseen_train_dev32_seed1403"
DEV112="$OFF/real_robotwin_eval/public_unseen_train_dev112_seed1403"
OUTPUT="$OFF/real_robotwin_eval/public_unseen_train_dev112_metrics_seed1403"
VARIANT='v203b_step8_seed1403'
SCREEN_PREREG="$REG/public_policy_screen_preregistration.json"
SUMMARY="$RUN/audit/public_policy_stage_a_summary.json"
AUDIT="$RUN/audit/public_policy_stage_a_acceptance.json"

test -s "$CHECKPOINT"
"$PY" - "$TRAIN_AUDIT" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
assert report["passed"] is True, report
PY
if [[ ! -e "$SCREEN_PREREG" ]]; then
  "$PY" "$BASE/pipeline/scripts/prepare_p3_public_policy_screen.py" \
    --checkpoint "$CHECKPOINT" \
    --training-audit "$TRAIN_AUDIT" \
    --training-preregistration "$TRAIN_PREREG" \
    --dev32-manifest "$DEV32/manifest.json" \
    --dev112-manifest "$DEV112/manifest.json" \
    --output "$SCREEN_PREREG" > "$REG/public_policy_screen_preregistration.log"
else
  "$PY" - "$SCREEN_PREREG" "$CHECKPOINT" <<'PY'
import hashlib, json, sys
from pathlib import Path
prereg, checkpoint = map(Path, sys.argv[1:])
payload = json.loads(prereg.read_text(encoding="utf-8"))
h = hashlib.sha256()
with checkpoint.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        h.update(block)
assert payload["candidate"]["checkpoint"] == str(checkpoint)
assert payload["candidate"]["checkpoint_sha256"] == h.hexdigest()
PY
fi

# Training and world-model inference have finished.  Release only the named
# processes owned by this pipeline before starting the GPU-heavy real simulator.
for session in wm_v202_bridge wm_v202_gpu; do
  if screen -ls 2>/dev/null | grep -q "[.]$session"; then
    screen -S "$session" -X quit || true
  fi
done
"$PY" -m ray.scripts.scripts stop --force > "$RUN/audit/ray_stop_before_public_stage_a.log" 2>&1 || true
for _ in $(seq 1 30); do
  if ! curl -fsS --max-time 1 http://127.0.0.1:8004/v1/health >/dev/null 2>&1 \
     && ! curl -fsS --max-time 1 http://127.0.0.1:18083/health >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

for batch in 00 01; do
  TRACK2_DEV_SEED_ROOT="$DEV112" \
  TRACK2_DEV_OUTPUT_ROOT="$OUTPUT" \
  TRACK2_CANDIDATE_CHECKPOINT="$CHECKPOINT" \
  TRACK2_CANDIDATE_VARIANT="$VARIANT" \
  TRACK2_DEV_BATCH_FILTER="$batch" \
    bash "$BASE/pipeline/scripts/run_strict_track2_dev_eval.sh"
done

"$PY" "$BASE/pipeline/scripts/summarize_strict_track2_dev_eval.py" \
  --output-root "$OUTPUT" \
  --dev-root "$DEV32" \
  --candidate "$VARIANT" \
  --output "$SUMMARY" > "$RUN/audit/public_policy_stage_a_summary.log"

set +e
"$PY" "$BASE/pipeline/scripts/audit_p3_public_policy_screen.py" \
  --preregistration "$SCREEN_PREREG" \
  --summary "$SUMMARY" \
  --stage stage_a \
  --output "$AUDIT" > "$RUN/audit/public_policy_stage_a_acceptance.log" 2>&1
audit_rc=$?
set -e
if (( audit_rc == 0 )); then
  touch "$RUN/audit/PUBLIC_POLICY_STAGE_A_PASSED"
  printf 'V203B_PUBLIC_POLICY_STAGE_A_PASSED\n'
else
  touch "$RUN/audit/PUBLIC_POLICY_STAGE_A_REJECTED"
  printf 'V203B_PUBLIC_POLICY_STAGE_A_REJECTED rc=%s\n' "$audit_rc"
fi
exit "$audit_rc"
