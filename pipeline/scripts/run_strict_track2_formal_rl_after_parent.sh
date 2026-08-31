#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-/root/autodl-tmp/IROS_WAM_2.0 challenge}
PARENT_PID=${2:-0}
JOINT="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
AUDIT="$ROOT/artifacts/strict_track2_official_20260810"
RUNTIME=/root/autodl-tmp/iros_v15_rl_probability_audit_v2
SCREEN="$JOINT/formal_v15_gated_full_composite_screen.json"
RELEASE="$JOINT/formal_v15_gated_onpolicy_release"
SERVICE_LOG="$JOINT/formal_v15_gated_service.log"
SERVICE_PID_FILE="$JOINT/formal_v15_gated_service.pid"
ACTOR_SEED=1236
CANDIDATE_VARIANT=gated_formal_seed1236
CHECKPOINT="$AUDIT/runs/official_replica_seed_${ACTOR_SEED}/wan_robotwin_adjust_bottle_http_grpo_openpi_pi05/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt"

if [[ "$PARENT_PID" =~ ^[1-9][0-9]*$ ]]; then
  while kill -0 "$PARENT_PID" 2>/dev/null; do sleep 10; done
fi
/root/autodl-tmp/conda_envs/rlinf_track2/bin/python - <<'PY' "$SCREEN"
import json,sys
report=json.load(open(sys.argv[1]))
assert report["passed"] is True, report
PY
test -s "$RELEASE/onpolicy_adaptation/adaptation_manifest.json"

mkdir -p "$JOINT/service_audit"
old_pid=""
for pid in $(pgrep -f 'pipeline/scripts/serve[.]py' || true); do
  if tr '\0' '\n' <"/proc/$pid/environ" 2>/dev/null | grep -Fqx 'WAM_PORT=8001'; then
    old_pid="$pid"
    tr '\0' '\n' <"/proc/$pid/environ" | \
      grep -E '^(WAM_BACKEND|WAM_CHECKPOINT_DIR|WAM_DEVICE|WAM_MODEL_VERSION|WAM_PORT|WAM_V15_LIBRARY_DIR)=' \
      >"$JOINT/service_audit/replaced_service_environment.txt"
    break
  fi
done
if [[ -n "$old_pid" ]]; then
  kill "$old_pid"
  for _ in $(seq 1 30); do kill -0 "$old_pid" 2>/dev/null || break; sleep 1; done
  if kill -0 "$old_pid" 2>/dev/null; then
    printf 'Old service did not stop cleanly: %s\n' "$old_pid" >&2
    exit 3
  fi
fi

cd "$RUNTIME"
nohup env \
  PYTHONPATH=pipeline \
  WAM_BACKEND=v15-gated-composite \
  WAM_BEARER_TOKEN=local-dev-token \
  WAM_CHECKPOINT_DIR="$RELEASE" \
  WAM_DEVICE=cpu \
  WAM_MODEL_VERSION=track2-v15.0-best \
  WAM_PORT=8001 \
  WAM_V15_LIBRARY_DIR="$ROOT/artifacts" \
  /root/miniconda3/envs/go1/bin/python pipeline/scripts/serve.py \
  >"$SERVICE_LOG" 2>&1 < /dev/null &
service_pid=$!
echo "$service_pid" >"$SERVICE_PID_FILE"
for _ in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:8001/v1/health | grep -q '"status":"ready"'; then break; fi
  sleep 1
done
curl -fsS http://127.0.0.1:8001/v1/health | grep -q '"model_version":"track2-v15.0-best"'
curl -fsS http://127.0.0.1:18080/health | grep -q '"status":"ready"'
sha256sum "$RELEASE/onpolicy_adaptation/adaptation_manifest.json" \
  >"$JOINT/service_audit/formal_release_manifest_sha256.txt"

"$AUDIT/run_strict_track2_official_replica.sh" "$ACTOR_SEED"
test -s "$CHECKPOINT"
TRACK2_CANDIDATE_CHECKPOINT="$CHECKPOINT" \
TRACK2_CANDIDATE_VARIANT="$CANDIDATE_VARIANT" \
  "$AUDIT/real_robotwin_eval/run_strict_track2_dev_eval.sh"
/root/autodl-tmp/conda_envs/rlinf_track2/bin/python \
  "$ROOT/pipeline/scripts/summarize_strict_track2_dev_eval.py" \
  --eval-root "$AUDIT/real_robotwin_eval/development_metrics" \
  --candidate-variant "$CANDIDATE_VARIANT" \
  --output "$AUDIT/real_robotwin_eval/dev_ranking_gated_formal_seed1236.json"
printf 'FORMAL_RL_DEVELOPMENT_COMPLETE checkpoint=%s\n' "$CHECKPOINT"
