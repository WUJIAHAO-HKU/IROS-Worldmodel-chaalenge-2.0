#!/usr/bin/env bash
set -euo pipefail
BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'; OFF="$BASE/artifacts/strict_track2_official_20260810"; P="$BASE/pipeline/scripts"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
NAME='v221_v169_rightsft_kl_h8_step24_seed1421_20260818'; RUN="$OFF/runs/$NAME"; REG="$OFF/run_registry/$NAME"
DEV="$OFF/real_robotwin_eval/public_unseen_train_dev112_seed1403"; OUT="$OFF/real_robotwin_eval/public_unseen_train_dev112_metrics_seed1403"
VARIANT='v221_v169_rightsft_kl_step24_seed1421'
CKPT="$RUN/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_24/actor/model_state_dict/full_weights.pt"
EVAL_LOG="$OUT/$VARIANT/batch_00/launcher.log"; PREREG="$REG/public_right_gate_preregistration.json"; REPORT="$RUN/audit/public_right_gate_acceptance.json"
test -f "$RUN/audit/V221_SFT_TRAINING_ACCEPTED"; test -f "$RUN/audit/policy_weight_delta.json"
test ! -e "$OUT/$VARIANT"; test ! -e "$PREREG"

"$PY" - "$PREREG" "$CKPT" "$DEV/manifest.json" "$DEV/batch_00.json" <<'PY'
import hashlib,json,pathlib,sys
from datetime import datetime,timezone
out,ckpt,manifest,batch=map(pathlib.Path,sys.argv[1:])
def h(p):
 d=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(8<<20),b''): d.update(c)
 return d.hexdigest()
out.write_text(json.dumps({
 'created_at':datetime.now(timezone.utc).isoformat(),
 'variant':'v221_v169_rightsft_kl_step24_seed1421',
 'purpose':'repeat the identical public batch00 right-arm gate after preregistered stronger training',
 'selection_uses_policy_outcomes':False,'batch':0,
 'episode_composition':{'left':4,'right':12,'total':16},
 'thresholds':{'right_success':1,'left_success':2,'total_success':3,'grasp_once':12},
 'checkpoint_sha256':h(ckpt),'manifest_sha256':h(manifest),'batch_sha256':h(batch),
 'real_submission':False,'reserved_final128_used':False,
 'on_reject':'do not evaluate batches 01-06; revise public-data training',
},indent=2)+'\n')
PY

restore(){ "$PY" -m ray.scripts.scripts stop --force > "$REG/ray_stop_after_public_right_gate.log" 2>&1 || true; bash "$P/restart_v218_services.sh" start > "$REG/restart_v218_after_public_right_gate.log" 2>&1 || true; }
trap restore EXIT
bash "$P/restart_v218_services.sh" stop
"$PY" -m ray.scripts.scripts stop --force > "$REG/ray_stop_before_public_right_gate.log" 2>&1 || true
TRACK2_DEV_SEED_ROOT="$DEV" TRACK2_DEV_OUTPUT_ROOT="$OUT" TRACK2_CANDIDATE_CHECKPOINT="$CKPT" \
TRACK2_CANDIDATE_VARIANT="$VARIANT" TRACK2_DEV_BATCH_FILTER=00 TRACK2_SKIP_BASELINE=true \
bash "$P/run_strict_track2_dev_eval.sh" > "$REG/public_right_gate_eval.log" 2>&1
set +e
"$PY" "$P/audit_v220_public_right_gate.py" "$PREREG" "$EVAL_LOG" "$REPORT" > "$RUN/audit/public_right_gate_acceptance.log" 2>&1
rc=$?
set -e
if ((rc)); then touch "$RUN/audit/PUBLIC_RIGHT_GATE_REJECTED"; exit "$rc"; fi
touch "$RUN/audit/PUBLIC_RIGHT_GATE_PASSED"; echo V221_PUBLIC_RIGHT_GATE_PASSED
