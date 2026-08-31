#!/usr/bin/env bash
set -euo pipefail

ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
O="$ROOT/artifacts/strict_track2_official_20260810"
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
P="$ROOT/pipeline/scripts"
RLINF="$ROOT/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
DIFFSYNTH="$O/official_deps/diffsynth_2a2e05f"
PY='/root/miniconda3/envs/go1/bin/python'
NAME='v432_public_mirror_prompt_terminal_seed1582_20260823'
REG="$O/run_registry/$NAME"
WORK="/dev/shm/$NAME"
SPLIT="$J/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json"
PARENT="$J/v354_v353_parametric_right_dynamics_extension_seed1523_20260822/model/checkpoints/checkpoint_step_000150"
ORIGINAL_STEP25="$WORK/model/checkpoint_step_000025"
STEP25_SHA256='dff072aff2f5c64261f9f968cd9ae436440132edbe06a6cf9e1bd2549cdadf7f'
REPLAY_ROOT="$WORK/replay50"
REPLAY25="$REPLAY_ROOT/checkpoint_step_000025"
REPLAY50="$REPLAY_ROOT/checkpoint_step_000050"
ACCEPTED_ROOT="$WORK/model_step50"
ACCEPTED50="$ACCEPTED_ROOT/checkpoint_step_000050"
REWARD="$ROOT/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt"
T5="$ROOT/artifacts/official_resources/reward_model/t5-base"
RESET="$ROOT/artifacts/rlinf_public_reset_adjust_bottle_train40/manifest.json"
AUTH="$REG/step50_continuation_authorization.json"
REPLAY_REPORT="$REG/replay_step25_verification.json"
GATE_REPORT="$REG/step50_recursive_gate.json"

cd "$ROOT"
test -s "$REG/preregistration.json"
test -s "$REG/step25_shortgate.json"
test -s "$ORIGINAL_STEP25/model.pt"
test "$(sha256sum "$ORIGINAL_STEP25/model.pt" | awk '{print $1}')" = "$STEP25_SHA256"
test -s "$PARENT/model.pt"
test -s "$REWARD"
test -d "$T5"
test ! -e "$AUTH"
test ! -e "$REPLAY_ROOT"
test ! -e "$ACCEPTED_ROOT"
test ! -e "$REPLAY_REPORT"
test ! -e "$GATE_REPORT"
"$PY" -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x["format"]=="strict-track2-v432-step25-shortgate-v1" and x["passed"] is True' "$REG/step25_shortgate.json"

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6
export PYTHONPATH="$ROOT/pipeline:$P:$RLINF:$DIFFSYNTH"

"$PY" "$P/prepare_v432_step50_continuation.py" \
  --original-preregistration "$REG/preregistration.json" \
  --step25-gate "$REG/step25_shortgate.json" --step25-checkpoint "$ORIGINAL_STEP25" \
  --trainer "$P/train_v432_step50_continuation.py" \
  --auditor "$P/audit_v432_step50_recursive_gate.py" \
  --launcher "$P/launch_v432_step50_continuation_and_recursive_gate.sh" \
  --output "$AUTH"

# Exact replay of the preregistered recipe: original parent, one AdamW, one
# seed-1582 sampling stream, checkpoints at global steps 25 and 50.
taskset -c 0-11 "$PY" "$P/train_v432_step50_continuation.py" \
  --windows "$ROOT/artifacts/adjust_bottle_windows_full" \
  --split-manifest "$SPLIT" --init-checkpoint "$PARENT" \
  --reward-checkpoint "$REWARD" --t5-model "$T5" --reset-manifest "$RESET" \
  --output "$REPLAY_ROOT" \
  --steps 50 --batch-size 1 --chunks 4 --chunk-stride 8 --arm-filter right \
  --learning-rate 1e-7 --reward-loss-weight 0.1 --reward-objective probability \
  --reward-probability-scale-floor 0.01 --reward-delta-weight 2.0 \
  --reward-terminal-weight 10.0 --right-weight 1.0 --success-weight 1.0 \
  --late-weight 1.0 --late-start 64 --terminal-visual-weight 2.0 \
  --prompts-per-arm 4 --checkpoint-interval 25 --max-grad-norm 1.0 \
  --seed 1582 --device cuda \
  >"$REG/step50_replay_training.log" 2>&1
test -s "$REPLAY25/model.pt"
test -s "$REPLAY50/model.pt"
test "$(find "$REPLAY_ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 2

# Always write the identity report.  A mismatch exits 2 here, before promotion
# or audit, while retaining both replay checkpoints for diagnosis.
"$PY" - "$STEP25_SHA256" "$REPLAY25/model.pt" "$REPLAY_REPORT" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

expected, model_name, report_name = sys.argv[1:]
model = Path(model_name)
observed = hashlib.sha256(model.read_bytes()).hexdigest()
passed = observed == expected
payload = {
    "format": "strict-track2-v432-replay-step25-identity-v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "passed": passed,
    "expected_model_sha256": expected,
    "observed_model_sha256": observed,
    "decision": "step50 may be promoted and audited" if passed else "stop before promotion and audit",
}
Path(report_name).write_text(json.dumps(payload, indent=2) + "\n")
raise SystemExit(0 if passed else 2)
PY

mkdir -p "$ACCEPTED_ROOT"
mv "$REPLAY50" "$ACCEPTED50"
rm -rf -- "$REPLAY25"
rmdir "$REPLAY_ROOT"
test -s "$ACCEPTED50/model.pt"
test "$(find "$ACCEPTED_ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 1
test "$(basename "$(find "$ACCEPTED_ROOT" -mindepth 1 -maxdepth 1 -type d)")" = 'checkpoint_step_000050'

# The auditor writes its full report before returning 2.  This launcher has no
# trace, RL, policy-update, hidden-data, or submission path.
taskset -c 0-11 "$PY" "$P/audit_v432_step50_recursive_gate.py" \
  --parent "$PARENT" --step25 "$ORIGINAL_STEP25" --step50 "$ACCEPTED50" \
  --windows "$ROOT/artifacts/adjust_bottle_windows_full" \
  --source-data "$ROOT/artifacts/datasets/aloha-agilex_clean_50/data" \
  --split "$SPLIT" --reward-checkpoint "$REWARD" --t5-model "$T5" \
  --original-preregistration "$REG/preregistration.json" \
  --step25-gate "$REG/step25_shortgate.json" --continuation-authorization "$AUTH" \
  --replay-verification "$REPLAY_REPORT" --output "$GATE_REPORT" \
  --device cuda --reward-batch-size 32 \
  >"$REG/step50_recursive_gate.log" 2>&1
