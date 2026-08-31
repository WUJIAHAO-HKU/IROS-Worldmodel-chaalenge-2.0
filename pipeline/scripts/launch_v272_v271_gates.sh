#!/usr/bin/env bash
set -euo pipefail
B='/root/autodl-tmp/IROS_WAM_2.0 challenge'
O="$B/artifacts/strict_track2_official_20260810"
J="$B/artifacts/strict_track2_joint_augmentation_20260810"
N='v272_v271_endpoint_calibrated_gates_seed1469_20260819'
RUN="$J/$N"; REG="$O/run_registry/$N"; S="$B/pipeline/scripts"
GO1='/root/miniconda3/envs/go1/bin/python'
RLPY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
RLINF="$B/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
V='track2-v271-endpoint-calibrated-terminal'; URL='http://127.0.0.1:8005'
PUBLIC="$B/artifacts/adjust_bottle_windows_full"
FAIL="$J/onpolicy_windows_full128_stride4"
BASELINE="$J/v212_v208_right_logit_long128_seed1411/audit/baseline"
REWARD="$B/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt"
T5="$B/artifacts/official_resources/reward_model/t5-base"
RESET="$B/artifacts/rlinf_public_reset_adjust_bottle_train40/manifest.json"
restore() { bash "$S/restart_v271_services.sh" stop >/dev/null 2>&1 || true; bash "$S/restart_v218_services.sh" start >"$REG/restart_v218.log" 2>&1 || true; }
trap restore EXIT
"$RLPY" "$S/prepare_v272_v271_gates.py"
ln -s "$BASELINE/public_success_long128.npz" "$RUN/audit/public_success_baseline.npz"
ln -s "$BASELINE/public_failure_long128.npz" "$RUN/audit/public_failure_baseline.npz"
bash "$S/restart_v271_services.sh" start >"$RUN/start_service.log" 2>&1
PYTHONPATH="$B/pipeline" "$GO1" "$S/strict_service_acceptance.py" --base-url "$URL" --token-file "$RUN/local_dev_token.txt" --model-version "$V" --output "$RUN/audit/service_acceptance.json" >"$RUN/audit/service_acceptance.log" 2>&1
set +e
cd /tmp
PYTHONPATH="$B/pipeline:$RLINF" TOKENIZERS_PARALLELISM=false "$RLPY" "$S/replay_v245_post_grasp_reward.py" --audit-dir "$O/run_registry/v211_v209_train_action_capture_h200_r2_step1_seed1410_20260818/bridge_audit" --library "$J/v214_public_right_knn_action_visual_diagnostic_seed1413/library/public_right_knn.npz" --url "$URL" --token local-dev-token --model-version "$V" --reward-checkpoint "$REWARD" --t5-model "$T5" --alpha-scale 8 --delta-regime --delta-clean-max .01 --delta-clean-alignment-floor .95 --delta-ood-min .8 --delta-ood-alignment-floor .5 --success-like-min 4 --group-std-min .005 --global-min .05 --group-min .40 --output "$RUN/audit/post_grasp_reward_report.json" --details "$RUN/audit/post_grasp_reward_details.npz" >"$RUN/audit/post_grasp_reward.log" 2>&1
capture_rc=$?
set -e
if (( capture_rc != 0 )); then touch "$RUN/V272_CAPTURE_GATE_REJECTED"; exit "$capture_rc"; fi
touch "$RUN/V272_CAPTURE_GATE_PASSED"
cd "$B"
PYTHONPATH="$B/pipeline:$S:$RLINF" "$GO1" "$S/export_v217_service_recursive_cache.py" --baseline-cache "$RUN/audit/public_success_baseline.npz" --query-windows "$PUBLIC" --url "$URL" --token local-dev-token --model-version "$V" --output "$RUN/audit/public_success_candidate.npz" --chunks 16 --batch-size 8 >"$RUN/audit/public_success_export.log" 2>&1
PYTHONPATH="$B/pipeline:$S:$RLINF" "$GO1" "$S/export_v217_service_recursive_cache.py" --baseline-cache "$RUN/audit/public_failure_baseline.npz" --query-windows "$FAIL" --url "$URL" --token local-dev-token --model-version "$V" --output "$RUN/audit/public_failure_candidate.npz" --chunks 16 --batch-size 8 >"$RUN/audit/public_failure_export.log" 2>&1
bash "$S/restart_v271_services.sh" stop
for _ in $(seq 1 30); do ! ss -ltn | grep -q ':8005 ' && break; sleep 1; done
PYTHONPATH="$B/pipeline:$S:$RLINF" "$GO1" "$S/evaluate_strict_track2_p2_reward_alignment.py" --cache "$RUN/audit/public_success_candidate.npz" --reuse-baseline-cache "$RUN/audit/public_success_baseline.npz" --reuse-baseline-report "$BASELINE/public_success_reward.json" --reward-checkpoint "$REWARD" --t5-model "$T5" --reset-manifest "$RESET" --output "$RUN/audit/public_success_reward.json" --batch-size 32 --device cuda >"$RUN/audit/public_success_reward.log" 2>&1
PYTHONPATH="$B/pipeline:$S:$RLINF" "$GO1" "$S/evaluate_strict_track2_p2_reward_alignment.py" --cache "$RUN/audit/public_failure_candidate.npz" --reuse-baseline-cache "$RUN/audit/public_failure_baseline.npz" --reuse-baseline-report "$BASELINE/public_failure_reward.json" --reward-checkpoint "$REWARD" --t5-model "$T5" --reset-manifest "$RESET" --output "$RUN/audit/public_failure_reward.json" --batch-size 32 --device cuda >"$RUN/audit/public_failure_reward.log" 2>&1
"$GO1" "$S/summarize_v256_long_gate.py" --run "$RUN" --registry "$REG" >"$RUN/audit/long_gate_report.log" 2>&1
"$GO1" - "$RUN/audit/long_gate_report.json" <<'PY'
import json, pathlib, sys
report = json.loads(pathlib.Path(sys.argv[1]).read_text())
raise SystemExit(0 if report["passed"] else 3)
PY
touch "$RUN/V272_LONG_GATE_PASSED" "$RUN/AUDIT_COMPLETE"
echo V272_V271_GATES_PASSED
