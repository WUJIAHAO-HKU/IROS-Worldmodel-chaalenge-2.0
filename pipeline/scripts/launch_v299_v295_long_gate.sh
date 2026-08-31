#!/usr/bin/env bash
set -euo pipefail
B='/root/autodl-tmp/IROS_WAM_2.0 challenge'; O="$B/artifacts/strict_track2_official_20260810"; J="$B/artifacts/strict_track2_joint_augmentation_20260810"
N='v299_v295_corrected_numeric_long_gate_seed1481_20260821'; RUN="$J/$N"; REG="$O/run_registry/$N"; S="$B/pipeline/scripts"
GO1='/root/miniconda3/envs/go1/bin/python'; RLPY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'; RLINF="$B/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
V='track2-v295-terminal-frame-preserving-mirror-v271'; URL='http://127.0.0.1:8005'; PUBLIC="$B/artifacts/adjust_bottle_windows_full"; FAIL="$J/onpolicy_windows_full128_stride4"
BASELINE="$J/v212_v208_right_logit_long128_seed1411/audit/baseline"; REWARD="$B/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt"; T5="$B/artifacts/official_resources/reward_model/t5-base"; RESET="$B/artifacts/rlinf_public_reset_adjust_bottle_train40/manifest.json"
PARENT="$J/v272_v271_endpoint_calibrated_gates_seed1469_20260819/audit"; CAPTURE="$J/v297_v295_terminal_frame_gates_seed1480_20260821/audit"
restore(){ bash "$S/restart_v295_services.sh" stop >/dev/null 2>&1 || true; TRACK2_CPUSET=0-23 bash "$S/restart_v271_v274_services.sh" start >"$REG/restart_v271_v274.log" 2>&1 || true; }
trap restore EXIT
"$RLPY" "$S/prepare_v299_v295_long_gate.py"
ln -s "$BASELINE/public_success_long128.npz" "$RUN/audit/public_success_baseline.npz"
ln -s "$BASELINE/public_failure_long128.npz" "$RUN/audit/public_failure_baseline.npz"
ln -s "$CAPTURE/post_grasp_reward_report.json" "$RUN/audit/post_grasp_reward_report.json"
ln -s "$CAPTURE/post_grasp_reward_details.npz" "$RUN/audit/post_grasp_reward_details.npz"
"$RLPY" "$S/verify_v298_capture_relative.py" --parent-report "$PARENT/post_grasp_reward_report.json" --parent-details "$PARENT/post_grasp_reward_details.npz" --candidate-report "$RUN/audit/post_grasp_reward_report.json" --candidate-details "$RUN/audit/post_grasp_reward_details.npz" --output "$RUN/audit/capture_relative_gate.json" >"$RUN/audit/capture_relative_gate.log" 2>&1
touch "$RUN/V299_CAPTURE_GATE_PASSED"
TRACK2_SERVICE_REG="$REG" TRACK2_CPUSET=0-23 bash "$S/restart_v295_services.sh" start >"$RUN/start_service.log" 2>&1
PYTHONPATH="$B/pipeline" "$GO1" "$S/strict_service_acceptance.py" --base-url "$URL" --token-file "$RUN/local_dev_token.txt" --model-version "$V" --output "$RUN/audit/service_acceptance.json" >"$RUN/audit/service_acceptance.log" 2>&1
cd "$B"
taskset -c 0-47 env PYTHONPATH="$B/pipeline:$S:$RLINF" "$GO1" "$S/export_v217_service_recursive_cache.py" --baseline-cache "$RUN/audit/public_success_baseline.npz" --query-windows "$PUBLIC" --url "$URL" --token local-dev-token --model-version "$V" --output "$RUN/audit/public_success_candidate.npz" --chunks 16 --batch-size 8 >"$RUN/audit/public_success_export.log" 2>&1
taskset -c 0-47 env PYTHONPATH="$B/pipeline:$S:$RLINF" "$GO1" "$S/export_v217_service_recursive_cache.py" --baseline-cache "$RUN/audit/public_failure_baseline.npz" --query-windows "$FAIL" --url "$URL" --token local-dev-token --model-version "$V" --output "$RUN/audit/public_failure_candidate.npz" --chunks 16 --batch-size 8 >"$RUN/audit/public_failure_export.log" 2>&1
bash "$S/restart_v295_services.sh" stop
for _ in $(seq 1 30); do ! ss -ltn | grep -q ':8005 ' && break; sleep 1; done
taskset -c 0-47 env PYTHONPATH="$B/pipeline:$S:$RLINF" "$GO1" "$S/evaluate_strict_track2_p2_reward_alignment.py" --cache "$RUN/audit/public_success_candidate.npz" --reuse-baseline-cache "$RUN/audit/public_success_baseline.npz" --reuse-baseline-report "$BASELINE/public_success_reward.json" --reward-checkpoint "$REWARD" --t5-model "$T5" --reset-manifest "$RESET" --output "$RUN/audit/public_success_reward.json" --batch-size 32 --device cuda >"$RUN/audit/public_success_reward.log" 2>&1
taskset -c 0-47 env PYTHONPATH="$B/pipeline:$S:$RLINF" "$GO1" "$S/evaluate_strict_track2_p2_reward_alignment.py" --cache "$RUN/audit/public_failure_candidate.npz" --reuse-baseline-cache "$RUN/audit/public_failure_baseline.npz" --reuse-baseline-report "$BASELINE/public_failure_reward.json" --reward-checkpoint "$REWARD" --t5-model "$T5" --reset-manifest "$RESET" --output "$RUN/audit/public_failure_reward.json" --batch-size 32 --device cuda >"$RUN/audit/public_failure_reward.log" 2>&1
"$GO1" "$S/summarize_v256_long_gate.py" --run "$RUN" --registry "$REG" >"$RUN/audit/long_gate_report.log" 2>&1
"$GO1" - "$RUN/audit/long_gate_report.json" <<'PY'
import json,pathlib,sys
x=json.loads(pathlib.Path(sys.argv[1]).read_text()); raise SystemExit(0 if x['passed'] else 3)
PY
touch "$RUN/V299_LONG_GATE_PASSED" "$RUN/AUDIT_COMPLETE"
echo V299_V295_LONG_GATE_PASSED
