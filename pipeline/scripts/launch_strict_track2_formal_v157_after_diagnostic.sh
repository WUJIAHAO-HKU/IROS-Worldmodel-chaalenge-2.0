#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRACK2_ROOT:-/root/autodl-tmp/IROS_WAM_2.0 challenge}"
GO1="${TRACK2_GO1_PYTHON:-/root/miniconda3/envs/go1/bin/python}"
RL_PY="${TRACK2_RL_PYTHON:-/root/autodl-tmp/conda_envs/rlinf_track2/bin/python}"
AUDIT="$ROOT/artifacts/strict_track2_official_20260810"
JOINT="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
DIAGNOSTIC_RUN_ID="${TRACK2_DIAGNOSTIC_RUN_ID:-probability_consistent_one_update_v157_seed1243}"
DIAGNOSTIC_RUN="$AUDIT/runs/$DIAGNOSTIC_RUN_ID"
DIAGNOSTIC_STEP_DIR="$DIAGNOSTIC_RUN/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_1"
DIAGNOSTIC_CHECKPOINT="$DIAGNOSTIC_STEP_DIR/actor/model_state_dict/full_weights.pt"
DIAGNOSTIC_DISPOSAL="$DIAGNOSTIC_RUN/audit/nonformal_checkpoint_disposal.json"
FORMAL_RUN="$AUDIT/runs/formal_full_budget_track2-v15.7-hybrid-action-gated-reward-safe-blend12_seed1244"
SERVICE_DIR="$AUDIT/formal_v157_service_seed1244"
BRIDGE_DIR="$AUDIT/formal_v157_bridge_seed1244"
RELEASE="$JOINT/v157_hybrid_gate_blend12_formal_release"
SCREEN="$JOINT/v157_reward_safe_parent_screen.json"
MODEL_VERSION="track2-v15.7-hybrid-action-gated-reward-safe-blend12"
STATE="$AUDIT/formal_v157_seed1244_automatic_launcher.log"
BASE_RUNTIME="$AUDIT/full_budget_gpu_cache_release_runtime"
EXPERIMENT_NAME="wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05"
FINAL_STEP_DIR="$FORMAL_RUN/$EXPERIMENT_NAME/checkpoints/global_step_1000"
FINAL_CHECKPOINT="$FINAL_STEP_DIR/actor/model_state_dict/full_weights.pt"
PREREG_ROOT="$AUDIT/real_robotwin_eval/preregistered_initial_window_joint"
DEV_ROOT="$PREREG_ROOT/development32"
ACCEPT_ROOT="$PREREG_ROOT/acceptance128"
DEV_OUTPUT="$AUDIT/real_robotwin_eval/formal_v157_seed1244_development32_metrics"
ACCEPT_OUTPUT="$AUDIT/real_robotwin_eval/formal_v157_seed1244_acceptance128_metrics"
CANDIDATE_VARIANT="formal_v157_seed1244_step1000"
DEV_REPORT="$AUDIT/real_robotwin_eval/formal_v157_seed1244_development32_summary.json"
ACCEPT_REPORT="$AUDIT/real_robotwin_eval/formal_v157_seed1244_acceptance128_summary.json"
FINAL_STATUS="$AUDIT/formal_v157_seed1244_track2_status.json"
TOKEN_FILE="$AUDIT/service_acceptance/secrets/bearer_token"
REAL_SERVICE_ACCEPTANCE="$AUDIT/service_acceptance/real_v157_protocol_runtime_opt_v2_before_formal_seed1244.json"
REAL_RESTART_BEFORE="$AUDIT/service_acceptance/real_v157_restart_runtime_opt_v2_before_formal_seed1244.json"
REAL_RESTART_AFTER="$AUDIT/service_acceptance/real_v157_restart_runtime_opt_v2_after_formal_seed1244.json"
REAL_RESTART_COMPARISON="$AUDIT/service_acceptance/real_v157_restart_comparison_runtime_opt_v2_formal_seed1244.json"
CACHE_UNCACHED_REFERENCE="$AUDIT/service_acceptance/v157_retrieval_cache_uncached_probe.json"
CACHE_WARM_PROBE="$AUDIT/service_acceptance/v157_retrieval_cache_warm_probe.json"
CACHE_CACHED_PROBE="$AUDIT/service_acceptance/v157_retrieval_cache_cached_probe.json"
CACHE_COMPARISON="$AUDIT/service_acceptance/v157_retrieval_cache_equivalence.json"
REAL_CONTEXT_SOURCE="$JOINT/v157_full_budget_service/bridge_audit/rollout_000002.npz"
DEADWORK_BASELINE="$AUDIT/service_acceptance/v157_pre_deadwork_target_cache_real_context_batch8.json"
DEADWORK_COLD_NPZ="$AUDIT/service_acceptance/v157_deadwork_target_cache_cold_real_context_batch8.npz"
DEADWORK_COLD="$AUDIT/service_acceptance/v157_deadwork_target_cache_cold_real_context_batch8.json"
DEADWORK_HOT_NPZ="$AUDIT/service_acceptance/v157_deadwork_target_cache_hot_real_context_batch8.npz"
DEADWORK_HOT="$AUDIT/service_acceptance/v157_deadwork_target_cache_hot_real_context_batch8.json"
DEADWORK_COMPARISON="$AUDIT/service_acceptance/v157_deadwork_target_cache_equivalence.json"

# Do not compete with the active diagnostic or its real-RoboTwin continuation.
while pgrep -f "train_embodied_agent.py.*${DIAGNOSTIC_RUN_ID}" >/dev/null \
  || pgrep -f 'continue_strict_track2_after_probability_consistent_one_update.sh' >/dev/null; do
  printf '%s waiting_for_diagnostic_and_dev22\n' "$(date -Iseconds)" >> "$STATE"
  sleep 60
done

# The diagnostic is informative only.  The formal run deliberately returns to
# the audited base runtime and never consumes its checkpoint.  "Full budget"
# below means the byte-verified public full-reference YAML; the organizer's
# held-out sample counts are not public, so this launcher cannot certify them.
test -s "$DIAGNOSTIC_RUN/audit/strict_track2_probability_consistent_one_update_preregistration.json"
test -s "$DIAGNOSTIC_CHECKPOINT"
grep -aq 'Global Step:    1/1' "$DIAGNOSTIC_RUN/launcher.log"
grep -aqE 'historical_dev22_complete|rejected_after_batch00' \
  "$DIAGNOSTIC_RUN/audit/automatic_continuation.log"
test -d "$RELEASE"
test -s "$TOKEN_FILE"
test "$(stat -c '%a' "$TOKEN_FILE")" = "600"
bearer_token="$(<"$TOKEN_FILE")"
# Pin the exact audited source boundary before committing the full budget.  The
# actor/rollout hashes contain only the preregistered allocator-cache release;
# OpenPi, GRPO and FSDP hashes are byte-identical to official commit
# 6f5a981b34232fe77812b818a6ad7a4e6b8728ac.
test "$(sha256sum "$BASE_RUNTIME/examples/embodiment/config/wan_robotwin_adjust_bottle_grpo_openpi_pi05.yaml" | cut -d' ' -f1)" = "97c4a6944e55ffb0f263e02d8ef8eb9cd176b821aee6d5776fd59b3df19136fb"
test "$(sha256sum "$BASE_RUNTIME/examples/embodiment/config/model/pi0_5.yaml" | cut -d' ' -f1)" = "2151df8492e58f023cbe969024ab0d7d0e2aad79eae976d79d2aaef6b4c75a45"
test "$(sha256sum "$BASE_RUNTIME/rlinf/models/embodiment/openpi/openpi_action_model.py" | cut -d' ' -f1)" = "12e071480e24d03ca1ded1258c28e9fd2435c75a77fef236fbbf646e900662de"
test "$(sha256sum "$BASE_RUNTIME/rlinf/algorithms/utils.py" | cut -d' ' -f1)" = "f56a5335fb4dc332a371f5f4e35be552907ed4aea761a68b2a056ca2c3ff11f1"
test "$(sha256sum "$BASE_RUNTIME/rlinf/hybrid_engines/fsdp/fsdp_model_manager.py" | cut -d' ' -f1)" = "3fb759832882f7ec61d73d55449925944b813342ff4bd4b817373d216267e5c1"
test "$(sha256sum "$BASE_RUNTIME/rlinf/workers/actor/fsdp_actor_worker.py" | cut -d' ' -f1)" = "24347c8d08e99fc273630f874ee5435e934a8fdfb028f0fb1c937c258edaeefb"
test "$(sha256sum "$BASE_RUNTIME/rlinf/workers/rollout/hf/huggingface_worker.py" | cut -d' ' -f1)" = "0b6948272e53cda6ae18d90946073c417d1f844d7a1a0de4902747d990dbe37d"
test "$(sha256sum "$BASE_RUNTIME/rlinf/envs/world_model/world_model_wan_http_env.py" | cut -d' ' -f1)" = "40ef65a82c39eaa0c6c28b1669d3185706c630dc4c1fd30b5e9f09f79e5557ca"
"$RL_PY" - <<'PY' "$SCREEN"
import json, sys
assert json.load(open(sys.argv[1]))["passed"] is True
PY

# This checkpoint was produced with the explicitly nonformal probability-mode
# diagnostic patch and must never seed the official run.  Once its historical
# dev22 consumer has exited, retain immutable provenance and reclaim the ~20 GB
# recovery payload before the long-budget run.  The target is an exact path,
# never a glob or unresolved variable.
if [[ -d "$DIAGNOSTIC_STEP_DIR" ]]; then
  test -s "$DIAGNOSTIC_CHECKPOINT"
  grep -aq 'Global Step:    1/1' "$DIAGNOSTIC_RUN/launcher.log"
  "$RL_PY" - <<'PY' "$DIAGNOSTIC_CHECKPOINT" "$DIAGNOSTIC_RUN/launcher.log" "$DIAGNOSTIC_DISPOSAL"
import datetime, hashlib, json, pathlib, sys
checkpoint, launcher, output = map(pathlib.Path, sys.argv[1:])
h = hashlib.sha256()
with checkpoint.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        h.update(chunk)
record = {
    "format": "strict-track2-nonformal-checkpoint-disposal-v1",
    "created_at": datetime.datetime.now().astimezone().isoformat(),
    "classification": "nonformal_probability_consistency_diagnostic",
    "checkpoint_path_before_disposal": str(checkpoint),
    "checkpoint_size_bytes": checkpoint.stat().st_size,
    "checkpoint_sha256": h.hexdigest(),
    "launcher_log": str(launcher),
    "formal_training_consumed_checkpoint": False,
    "reason": "Historical dev22 consumer finished; reclaim storage before the independent public-reference seed1244 run.",
}
output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  case "$DIAGNOSTIC_STEP_DIR" in
    "$DIAGNOSTIC_RUN"/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_1)
      rm -rf -- "$DIAGNOSTIC_STEP_DIR"
      ;;
    *)
      printf '%s refusing_nonformal_disposal_target=%s\n' "$(date -Iseconds)" "$DIAGNOSTIC_STEP_DIR" >> "$STATE"
      exit 8
      ;;
  esac
  printf '%s disposed_nonformal_diagnostic_checkpoint provenance=%s\n' \
    "$(date -Iseconds)" "$DIAGNOSTIC_DISPOSAL" >> "$STATE"
fi

mkdir -p "$SERVICE_DIR" "$BRIDGE_DIR/audit"

service_pid=""
bridge_pid=""
cleanup() {
  if [[ -n "$bridge_pid" ]]; then kill "$bridge_pid" 2>/dev/null || true; fi
  if [[ -n "$service_pid" ]]; then kill "$service_pid" 2>/dev/null || true; fi
  "$RL_PY" -m ray.scripts.scripts stop --force > "$AUDIT/formal_v157_seed1244_ray_stop.log" 2>&1 || true
}
trap cleanup EXIT

stop_wm_stack() {
  if [[ -n "$bridge_pid" ]]; then
    kill "$bridge_pid" 2>/dev/null || true
    wait "$bridge_pid" 2>/dev/null || true
    bridge_pid=""
  fi
  if [[ -n "$service_pid" ]]; then
    kill "$service_pid" 2>/dev/null || true
    wait "$service_pid" 2>/dev/null || true
    service_pid=""
  fi
  "$RL_PY" -m ray.scripts.scripts stop --force > "$AUDIT/formal_v157_seed1244_ray_stop_before_real_eval.log" 2>&1 || true
}

start_wm_stack() {
  cd "$ROOT"
  PYTHONPATH="$ROOT/pipeline" WAM_MODEL_VERSION="$MODEL_VERSION" WAM_BEARER_TOKEN="$bearer_token" \
  WAM_BACKEND="v15-gated-composite" WAM_CHECKPOINT_DIR="$RELEASE" WAM_V15_LIBRARY_DIR="$ROOT/artifacts" \
  WAM_DEVICE="cuda" WAM_PORT=8001 WAM_NATIVE_BATCH_ENABLED=1 WAM_NATIVE_BATCH_MICRO_SIZE=8 \
  WAM_RELEASE_CUDA_CACHE=1 WAM_RETRIEVAL_SOURCE_CACHE=1 WAM_RETRIEVAL_TARGET_CACHE=1 \
  WAM_IMAGE_CODEC_WORKERS=8 \
  "$GO1" pipeline/scripts/serve.py >> "$SERVICE_DIR/service.log" 2>&1 &
  service_pid=$!
  for _ in $(seq 1 120); do
    if curl -fsS http://127.0.0.1:8001/v1/health 2>/dev/null | grep -q "$MODEL_VERSION"; then break; fi
    sleep 1
  done
  curl -fsS http://127.0.0.1:8001/v1/health | grep -q "$MODEL_VERSION"

  PYTHONPATH="$ROOT/pipeline" WAM_BEARER_TOKEN="$bearer_token" WAM_IMAGE_CODEC_WORKERS=8 \
    "$GO1" -m wam_pipeline.rlinf_bridge.server \
    --world-model-url http://127.0.0.1:8001 --model-version "$MODEL_VERSION" \
    --host 127.0.0.1 --port 18080 --audit-dir "$BRIDGE_DIR/audit" --audit-max-items 1 \
    >> "$BRIDGE_DIR/bridge.log" 2>&1 &
  bridge_pid=$!
  for _ in $(seq 1 60); do
    if curl -fsS http://127.0.0.1:18080/health 2>/dev/null | grep -q '"status":"ready"'; then break; fi
    sleep 1
  done
  curl -fsS http://127.0.0.1:18080/health | grep -q '"status":"ready"'
}

start_wm_stack

# The running pre-optimization service captured this batch before the source
# file was replaced.  Its first eight real rollout samples are all right-arm,
# complementing the synthetic left/tied-arm restart probe below.  Demand exact
# equality both on a cold lazy-target cache and on a second backend execution
# after its selected target windows are hot; request IDs are fresh so the HTTP
# idempotency cache cannot satisfy either call.
test -s "$REAL_CONTEXT_SOURCE"
test -s "$DEADWORK_BASELINE"
if [[ ! -s "$DEADWORK_COMPARISON" ]]; then
  printf '%s starting_v141_deadwork_target_cache_equivalence\n' "$(date -Iseconds)" >> "$STATE"
  # A power loss between the atomic NPZ and JSON renames can leave exactly one
  # member of a pair.  Such a pair is not admissible evidence; remove only
  # these exact generated paths and repeat the stateless probe.
  for pair in "$DEADWORK_COLD_NPZ|$DEADWORK_COLD" "$DEADWORK_HOT_NPZ|$DEADWORK_HOT"; do
    npz=${pair%%|*}; manifest=${pair#*|}
    if [[ -e "$npz" && ! -s "$manifest" ]] || [[ ! -s "$npz" && -e "$manifest" ]]; then
      printf '%s removing_incomplete_equivalence_pair npz=%s manifest=%s\n' \
        "$(date -Iseconds)" "$npz" "$manifest" >> "$STATE"
      rm -f -- "$npz" "$manifest"
    fi
  done
  if [[ ! -s "$DEADWORK_COLD_NPZ" || ! -s "$DEADWORK_COLD" ]]; then
    PYTHONPATH="$ROOT/pipeline" WAM_IMAGE_CODEC_WORKERS=8 "$GO1" \
      "$ROOT/pipeline/scripts/capture_strict_track2_service_golden_batch.py" \
      --golden "$REAL_CONTEXT_SOURCE" --url http://127.0.0.1:8001 \
      --token "$bearer_token" --model-version "$MODEL_VERSION" --begin 0 --count 8 \
      --output-npz "$DEADWORK_COLD_NPZ" --output-json "$DEADWORK_COLD" >> "$STATE"
  fi
  if [[ ! -s "$DEADWORK_HOT_NPZ" || ! -s "$DEADWORK_HOT" ]]; then
    PYTHONPATH="$ROOT/pipeline" WAM_IMAGE_CODEC_WORKERS=8 "$GO1" \
      "$ROOT/pipeline/scripts/capture_strict_track2_service_golden_batch.py" \
      --golden "$REAL_CONTEXT_SOURCE" --url http://127.0.0.1:8001 \
      --token "$bearer_token" --model-version "$MODEL_VERSION" --begin 0 --count 8 \
      --output-npz "$DEADWORK_HOT_NPZ" --output-json "$DEADWORK_HOT" >> "$STATE"
  fi
  "$GO1" - <<'PY' "$DEADWORK_BASELINE" "$DEADWORK_COLD" "$DEADWORK_HOT" "$DEADWORK_COMPARISON" "$ROOT/pipeline/wam_pipeline/v15_runtime.py"
import datetime, hashlib, json, pathlib, sys
import numpy as np
baseline_path, cold_path, hot_path, output_path, source_path = map(pathlib.Path, sys.argv[1:])
baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
cold = json.loads(cold_path.read_text(encoding="utf-8"))
hot = json.loads(hot_path.read_text(encoding="utf-8"))

def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def capture_self_consistent(row):
    path = pathlib.Path(row["capture_npz"])
    if not path.is_file() or file_sha(path) != row["capture_npz_sha256"]:
        return False
    with np.load(path, allow_pickle=False) as payload:
        predicted = payload["predicted_frames"]
    return (
        list(predicted.shape) == row["prediction_shape"]
        and str(predicted.dtype) == row["prediction_dtype"]
        and hashlib.sha256(np.ascontiguousarray(predicted).tobytes()).hexdigest()
        == row["prediction_sha256"]
    )

checks = {
    "capture_artifacts_self_consistent": all(
        capture_self_consistent(row) for row in (baseline, cold, hot)
    ),
    "same_real_context_source": len({row["source_npz_sha256"] for row in (baseline, cold, hot)}) == 1,
    "real_context_source_file_exact": file_sha(pathlib.Path(baseline["source_npz"]))
    == baseline["source_npz_sha256"],
    "same_model_version": len({row["model_version"] for row in (baseline, cold, hot)}) == 1,
    "same_slice": all(row["slice_begin"] == 0 and row["slice_count"] == 8 for row in (baseline, cold, hot)),
    "cold_pixels_exact": baseline["prediction_sha256"] == cold["prediction_sha256"],
    "hot_pixels_exact": baseline["prediction_sha256"] == hot["prediction_sha256"],
    "cold_hot_pixels_exact": cold["prediction_sha256"] == hot["prediction_sha256"],
}
runtime_sources = {
    "pipeline/wam_pipeline/v15_runtime.py": source_path,
    "pipeline/wam_pipeline/images.py": source_path.parent / "images.py",
    "pipeline/wam_pipeline/service.py": source_path.parent / "service.py",
    "pipeline/wam_pipeline/rlinf_bridge/track2_client.py": source_path.parent / "rlinf_bridge" / "track2_client.py",
}
runtime_source_sha256 = {
    relative: hashlib.sha256(path.read_bytes()).hexdigest()
    for relative, path in runtime_sources.items()
}
result = {
    "format": "strict-track2-v157-deadwork-target-cache-codec-equivalence-v2",
    "created_at": datetime.datetime.now().astimezone().isoformat(),
    "passed": all(checks.values()),
    "checks": checks,
    "baseline": str(baseline_path),
    "cold": str(cold_path),
    "hot": str(hot_path),
    "v15_runtime_sha256": runtime_source_sha256["pipeline/wam_pipeline/v15_runtime.py"],
    "runtime_source_sha256": runtime_source_sha256,
    "baseline_wall_seconds": baseline["wall_seconds_including_overload_retries"],
    "cold_wall_seconds": cold["wall_seconds_including_overload_retries"],
    "hot_wall_seconds": hot["wall_seconds_including_overload_retries"],
    "baseline_to_hot_speedup": baseline["wall_seconds_including_overload_retries"] / hot["wall_seconds_including_overload_retries"],
    "prediction_semantics_changed": False,
    "resource_changes": [
        "evaluate immutable v14.1 rejection predicates before unused ECC",
        "retain exact selected target uint8 arrays lazily in host RAM",
        "encode and decode independent PNG frames in stable order on eight CPU workers",
    ],
}
temporary = output_path.with_suffix(output_path.suffix + ".tmp")
temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(output_path)
assert result["passed"]
PY
  printf '%s v141_deadwork_target_cache_equivalence_passed\n' "$(date -Iseconds)" >> "$STATE"
else
  "$GO1" - <<'PY' "$DEADWORK_COMPARISON" "$ROOT"
import hashlib, json, pathlib, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
root = pathlib.Path(sys.argv[2])
assert report["passed"] is True
for relative, expected in report["runtime_source_sha256"].items():
    assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
PY
fi

# The source-frame cache retains bytes that V15 already decodes while loading
# its retrieval descriptors.  Compare a warmed cached service against the
# fixed uncached real-service probe before it is allowed into the formal run.
test -s "$CACHE_UNCACHED_REFERENCE"
if [[ ! -s "$CACHE_COMPARISON" ]]; then
  printf '%s starting_retrieval_source_cache_equivalence\n' "$(date -Iseconds)" >> "$STATE"
  PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" "$GO1" \
    "$ROOT/pipeline/scripts/strict_service_restart_probe.py" \
    --base-url http://127.0.0.1:8001 --token-file "$TOKEN_FILE" \
    --model-version "$MODEL_VERSION" \
    --request-id 00000000-0000-4000-8000-000000000002 \
    --output "$CACHE_WARM_PROBE" >> "$STATE"
  PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" "$GO1" \
    "$ROOT/pipeline/scripts/strict_service_restart_probe.py" \
    --base-url http://127.0.0.1:8001 --token-file "$TOKEN_FILE" \
    --model-version "$MODEL_VERSION" \
    --request-id 00000000-0000-4000-8000-000000000001 \
    --output "$CACHE_CACHED_PROBE" >> "$STATE"
  "$GO1" - <<'PY' "$CACHE_UNCACHED_REFERENCE" "$CACHE_CACHED_PROBE" "$CACHE_COMPARISON"
import datetime, json, pathlib, sys
uncached_path, cached_path, output_path = map(pathlib.Path, sys.argv[1:])
uncached = json.loads(uncached_path.read_text(encoding="utf-8"))
cached = json.loads(cached_path.read_text(encoding="utf-8"))
checks = {
    "capabilities_exact": uncached["capabilities_sha256"] == cached["capabilities_sha256"],
    "pixels_exact": uncached["pixel_sha256"] == cached["pixel_sha256"],
    "request_exact": uncached["request_id"] == cached["request_id"],
}
result = {
    "format": "strict-track2-v157-retrieval-source-cache-equivalence-v1",
    "created_at": datetime.datetime.now().astimezone().isoformat(),
    "passed": all(checks.values()),
    "checks": checks,
    "uncached_probe": str(uncached_path),
    "cached_probe": str(cached_path),
    "uncached_latency_ms": uncached["latency_ms"],
    "cached_warm_latency_ms": cached["latency_ms"],
    "speedup": uncached["latency_ms"] / cached["latency_ms"],
    "prediction_semantics_changed": False,
    "resource_change": "retain exact retrieval source uint8 arrays already decoded at startup",
}
output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
assert result["passed"]
PY
  printf '%s retrieval_source_cache_equivalence_passed\n' "$(date -Iseconds)" >> "$STATE"
else
  "$GO1" - <<'PY' "$CACHE_COMPARISON"
import json, sys
assert json.load(open(sys.argv[1], encoding="utf-8"))["passed"] is True
PY
fi

# Run the complete contract on the actual frozen V15.7 backend once, then
# restart both private processes and demand byte-identical capabilities and
# decoded output pixels.  A passed comparison is reusable across a power-cycle.
if ! "$GO1" - <<'PY' "$REAL_RESTART_COMPARISON" 2>/dev/null
import json, sys
assert json.load(open(sys.argv[1], encoding="utf-8"))["passed"] is True
PY
then
  printf '%s starting_real_v157_service_acceptance\n' "$(date -Iseconds)" >> "$STATE"
  PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" "$GO1" \
    "$ROOT/pipeline/scripts/strict_service_acceptance.py" \
    --base-url http://127.0.0.1:8001 --token-file "$TOKEN_FILE" \
    --model-version "$MODEL_VERSION" --output "$REAL_SERVICE_ACCEPTANCE" >> "$STATE"
  PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" "$GO1" \
    "$ROOT/pipeline/scripts/strict_service_restart_probe.py" \
    --base-url http://127.0.0.1:8001 --token-file "$TOKEN_FILE" \
    --model-version "$MODEL_VERSION" --output "$REAL_RESTART_BEFORE" >> "$STATE"
  stop_wm_stack
  start_wm_stack
  PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" "$GO1" \
    "$ROOT/pipeline/scripts/strict_service_restart_probe.py" \
    --base-url http://127.0.0.1:8001 --token-file "$TOKEN_FILE" \
    --model-version "$MODEL_VERSION" --output "$REAL_RESTART_AFTER" >> "$STATE"
  "$GO1" - <<'PY' "$REAL_SERVICE_ACCEPTANCE" "$REAL_RESTART_BEFORE" "$REAL_RESTART_AFTER" "$REAL_RESTART_COMPARISON"
import datetime, json, pathlib, sys
acceptance_path, before_path, after_path, output_path = map(pathlib.Path, sys.argv[1:])
acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
before = json.loads(before_path.read_text(encoding="utf-8"))
after = json.loads(after_path.read_text(encoding="utf-8"))
passed = (
    acceptance["passed"] is True
    and acceptance["latency_ms"]["all_under_recommended_timeout"] is True
    and before["capabilities_sha256"] == after["capabilities_sha256"]
    and before["pixel_sha256"] == after["pixel_sha256"]
)
result = {
    "format": "strict-track2-real-v15-restart-comparison-v1",
    "created_at": datetime.datetime.now().astimezone().isoformat(),
    "passed": passed,
    "protocol_acceptance": str(acceptance_path),
    "capabilities_sha256_before": before["capabilities_sha256"],
    "capabilities_sha256_after": after["capabilities_sha256"],
    "pixel_sha256_before": before["pixel_sha256"],
    "pixel_sha256_after": after["pixel_sha256"],
    "latency_ms_before": before["latency_ms"],
    "latency_ms_after": after["latency_ms"],
}
output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
assert passed
PY
  printf '%s real_v157_service_acceptance_passed\n' "$(date -Iseconds)" >> "$STATE"
else
  printf '%s real_v157_service_acceptance_already_passed\n' "$(date -Iseconds)" >> "$STATE"
fi

printf '%s starting_formal_unmodified_flow_public_full_reference_budget organizer_hidden_budget_not_claimed\n' "$(date -Iseconds)" >> "$STATE"
TRACK2_ACTOR_SEED=1244 TRACK2_PARENT_RELEASE="$RELEASE" TRACK2_PARENT_SCREEN="$SCREEN" \
TRACK2_BEARER_TOKEN="$bearer_token" \
TRACK2_MODEL_VERSION="$MODEL_VERSION" TRACK2_RUN_ROOT="$FORMAL_RUN" \
  "$ROOT/pipeline/scripts/run_strict_track2_full_budget_formal_offloaded.sh"
test -s "$FINAL_CHECKPOINT"
printf '%s formal_full_budget_complete\n' "$(date -Iseconds)" >> "$STATE"

# Real RoboTwin needs the GPU occupied by the world-model service.  Stop only
# this launcher's private service/bridge before evaluating the frozen policy.
stop_wm_stack

# The 32 development and 128 acceptance seeds were preregistered before this
# policy was trained and are disjoint.  Acceptance remains unopened unless the
# candidate first clears the paired dual-arm development gate.
printf '%s starting_preregistered_development32\n' "$(date -Iseconds)" >> "$STATE"
TRACK2_DEV_SEED_ROOT="$DEV_ROOT" TRACK2_DEV_OUTPUT_ROOT="$DEV_OUTPUT" \
TRACK2_CANDIDATE_CHECKPOINT="$FINAL_CHECKPOINT" TRACK2_CANDIDATE_VARIANT="$CANDIDATE_VARIANT" \
  "$ROOT/pipeline/scripts/run_strict_track2_dev_eval.sh" >> "$STATE" 2>&1
"$RL_PY" "$ROOT/pipeline/scripts/summarize_strict_track2_dev_eval.py" \
  --output-root "$DEV_OUTPUT" --dev-root "$DEV_ROOT" --candidate "$CANDIDATE_VARIANT" \
  --output "$DEV_REPORT" >> "$STATE"
"$RL_PY" - <<'PY' "$DEV_REPORT" "$CANDIDATE_VARIANT"
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
assert report["seed_count"] == 32
assert report["selected_variant"] == sys.argv[2]
PY
printf '%s development32_passed_starting_sealed_acceptance128\n' "$(date -Iseconds)" >> "$STATE"

TRACK2_DEV_SEED_ROOT="$ACCEPT_ROOT" TRACK2_DEV_OUTPUT_ROOT="$ACCEPT_OUTPUT" \
TRACK2_CANDIDATE_CHECKPOINT="$FINAL_CHECKPOINT" TRACK2_CANDIDATE_VARIANT="$CANDIDATE_VARIANT" \
  "$ROOT/pipeline/scripts/run_strict_track2_dev_eval.sh" >> "$STATE" 2>&1
"$RL_PY" "$ROOT/pipeline/scripts/summarize_strict_track2_dev_eval.py" \
  --output-root "$ACCEPT_OUTPUT" --dev-root "$ACCEPT_ROOT" --candidate "$CANDIDATE_VARIANT" \
  --output "$ACCEPT_REPORT" >> "$STATE"
"$RL_PY" - <<'PY' "$ACCEPT_REPORT" "$CANDIDATE_VARIANT" "$FINAL_CHECKPOINT" "$FINAL_STATUS"
import datetime, hashlib, json, pathlib, sys
report_path, variant, checkpoint_path, status_path = map(pathlib.Path, sys.argv[1:])
report = json.loads(report_path.read_text(encoding="utf-8"))
assert report["seed_count"] == 128
candidate = next(row for row in report["candidates"] if row["variant"] == str(variant))
comparison = candidate["comparison"]
passed = report["selected_variant"] == str(variant) and comparison["eligible"]
h = hashlib.sha256()
with checkpoint_path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        h.update(chunk)
status = {
    "format": "strict-track2-formal-policy-status-v1",
    "created_at": datetime.datetime.now().astimezone().isoformat(),
    "candidate_variant": str(variant),
    "checkpoint": str(checkpoint_path),
    "checkpoint_sha256": h.hexdigest(),
    "development32_passed": True,
    "acceptance128_passed": passed,
    "acceptance_success_delta": comparison["success_delta"],
    "acceptance_grasp_delta": comparison["grasp_delta"],
    "goal_complete": False,
    "remaining_gate": "real V15 HTTPS/auth/restart/latency acceptance" if passed else "policy acceptance >=3%",
}
pathlib.Path(status_path).write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
assert passed
PY
printf '%s acceptance128_passed_policy_gate_complete\n' "$(date -Iseconds)" >> "$STATE"
