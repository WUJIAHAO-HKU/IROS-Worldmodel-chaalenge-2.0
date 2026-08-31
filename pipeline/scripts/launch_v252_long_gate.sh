#!/usr/bin/env bash
set -euo pipefail
B='/root/autodl-tmp/IROS_WAM_2.0 challenge'; O="$B/artifacts/strict_track2_official_20260810"; J="$B/artifacts/strict_track2_joint_augmentation_20260810"; N='v252_v250_specific_terminal_long128_seed1455_20260819'; RUN="$J/$N"; REG="$O/run_registry/$N"; P="$B/pipeline/scripts"; PY='/root/miniconda3/envs/go1/bin/python'; V='track2-v250-specific-terminal-successor'; URL='http://127.0.0.1:8005'; PUBLIC="$B/artifacts/adjust_bottle_windows_full"; FAIL="$J/onpolicy_windows_full128_stride4"; BASELINE="$J/v212_v208_right_logit_long128_seed1411/audit/baseline"; REWARD="$B/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt"; T5="$B/artifacts/official_resources/reward_model/t5-base"; RESET="$B/artifacts/rlinf_public_reset_adjust_bottle_train40/manifest.json"; RLINF="$B/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"; export PYTHONPATH="$B/pipeline:$P:$RLINF"
restore(){ bash "$P/restart_v252_services.sh" stop >/dev/null 2>&1 || true; bash "$P/restart_v218_services.sh" start >"$REG/restart_v218.log" 2>&1 || true; }
trap restore EXIT
"$PY" "$P/prepare_v252_long_gate.py"
ln -s "$BASELINE/public_success_long128.npz" "$RUN/audit/public_success_baseline.npz"
ln -s "$BASELINE/public_failure_long128.npz" "$RUN/audit/public_failure_baseline.npz"
bash "$P/restart_v252_services.sh" start >"$RUN/start_service.log" 2>&1
"$PY" "$P/strict_service_acceptance.py" --base-url "$URL" --token-file "$RUN/local_dev_token.txt" --model-version "$V" --output "$RUN/audit/service_acceptance.json" >"$RUN/audit/service_acceptance.log" 2>&1
"$PY" "$P/export_v217_service_recursive_cache.py" --baseline-cache "$RUN/audit/public_success_baseline.npz" --query-windows "$PUBLIC" --url "$URL" --token local-dev-token --model-version "$V" --output "$RUN/audit/public_success_candidate.npz" --chunks 16 --batch-size 8 >"$RUN/audit/public_success_export.log" 2>&1
"$PY" "$P/export_v217_service_recursive_cache.py" --baseline-cache "$RUN/audit/public_failure_baseline.npz" --query-windows "$FAIL" --url "$URL" --token local-dev-token --model-version "$V" --output "$RUN/audit/public_failure_candidate.npz" --chunks 16 --batch-size 8 >"$RUN/audit/public_failure_export.log" 2>&1
bash "$P/restart_v252_services.sh" stop
for _ in $(seq 1 30); do ! ss -ltn | grep -q ':8005 ' && break; sleep 1; done
"$PY" "$P/evaluate_strict_track2_p2_reward_alignment.py" --cache "$RUN/audit/public_success_candidate.npz" --reuse-baseline-cache "$RUN/audit/public_success_baseline.npz" --reuse-baseline-report "$BASELINE/public_success_reward.json" --reward-checkpoint "$REWARD" --t5-model "$T5" --reset-manifest "$RESET" --output "$RUN/audit/public_success_reward.json" --batch-size 32 --device cuda >"$RUN/audit/public_success_reward.log" 2>&1
"$PY" "$P/evaluate_strict_track2_p2_reward_alignment.py" --cache "$RUN/audit/public_failure_candidate.npz" --reuse-baseline-cache "$RUN/audit/public_failure_baseline.npz" --reuse-baseline-report "$BASELINE/public_failure_reward.json" --reward-checkpoint "$REWARD" --t5-model "$T5" --reset-manifest "$RESET" --output "$RUN/audit/public_failure_reward.json" --batch-size 32 --device cuda >"$RUN/audit/public_failure_reward.log" 2>&1
"$PY" "$P/summarize_v252_long_gate.py" --run "$RUN" --registry "$REG" >"$RUN/audit/long_gate_report.log" 2>&1
touch "$RUN/AUDIT_COMPLETE"
echo V252_LONG_GATE_COMPLETE
