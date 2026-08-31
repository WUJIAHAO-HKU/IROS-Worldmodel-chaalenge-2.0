#!/usr/bin/env bash
set -euo pipefail

ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$ROOT/artifacts/strict_track2_official_20260810"
SOURCE_REG="$OFF/run_registry/v411_v169step5_v409_oneupdate_h200_r4_lr1e5_seed1569_20260823"
CKPT='/dev/shm/v411_v169step5_v409_oneupdate_h200_r4_lr1e5_seed1569_20260823/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt'
DEV="$OFF/real_robotwin_eval/public_unseen_train_dev112_seed1403"
OUT="$OFF/real_robotwin_eval/public_unseen_train_dev112_metrics_seed1403"
REG="$OFF/run_registry/v412_v411_public_batch00_20260823"
VARIANT='v412_v411_oneupdate_public_batch00'
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
PREREG="$REG/preregistration.json"
REPORT="$REG/result.json"
DRIVER_LOG="$REG/driver.log"
EXPECTED_SHA='8ff49ff7f947e2019096d1b09bdadb9c290e83632d8bcb88684526d327101e5d'

test ! -e "$REG"
mkdir -p "$REG"
test -s "$CKPT"
test -s "$SOURCE_REG/training_acceptance.json"
test -f "$SOURCE_REG/V411_ONE_UPDATE_ACCEPTED"
test "$(awk '{print $1}' "$SOURCE_REG/volatile_checkpoint.sha256")" = "$EXPECTED_SHA"
test -s "$DEV/manifest.json"
test -s "$DEV/batch_00.json"
test ! -e "$OUT/$VARIANT/batch_00/launcher.log"

"$PY" - "$CKPT" "$DEV/manifest.json" "$DEV/batch_00.json" \
  "$SOURCE_REG/training_acceptance.json" "$PREREG" "$EXPECTED_SHA" <<'PY'
import datetime, hashlib, json, pathlib, sys

checkpoint, manifest, batch00, training, output = map(pathlib.Path, sys.argv[1:6])
expected_sha = sys.argv[6]

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

training_payload = json.loads(training.read_text())
assert training_payload["passed"] is True
assert expected_sha == sha256(checkpoint)
payload = {
    "format": "strict-track2-v412-frozen-candidate-public-batch00-preregistration-v1",
    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "variant": "v412_v411_oneupdate_public_batch00",
    "purpose": "First outcome-based go/no-go for the uniquely frozen v411 policy; target-aligned pilot before public112.",
    "candidate_frozen_before_this_preregistration": True,
    "training_stability_passed": True,
    "selection_uses_reserved_final128": False,
    "real_submission": False,
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": expected_sha,
    "training_acceptance": str(training),
    "training_acceptance_sha256": sha256(training),
    "public_manifest": str(manifest),
    "public_manifest_sha256": sha256(manifest),
    "batch00": str(batch00),
    "batch00_sha256": sha256(batch00),
    "batch00_composition": {"total": 16, "left": 4, "right": 12},
    "acceptance": {
        "total_success_at_least": 10,
        "left_success_at_least": 3,
        "right_success_at_least": 7,
        "grasp_once_at_least": 14,
    },
    "target_alignment": {
        "final_goal": "at least 85/128",
        "public112_gate": "at least 75/112",
        "batch00_total_floor": "10/16; deliberately stricter than the retired 6/16 diagnostic floor",
    },
    "on_reject": "Stop v411 without accessing public batches 01-06 or reserved final128.",
    "on_accept": "Evaluate unchanged checkpoint on public batches 01-06 and require at least 75/112 total with both arms represented.",
}
output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

restore() {
  "$PY" -m ray.scripts.scripts stop --force >"$REG/ray_stop_after.log" 2>&1 || true
  bash "$ROOT/pipeline/scripts/restart_v218_services.sh" start >"$REG/restore_v218.log" 2>&1 || true
}
trap restore EXIT

bash "$ROOT/pipeline/scripts/restart_v218_services.sh" stop
"$PY" -m ray.scripts.scripts stop --force >"$REG/ray_stop_before.log" 2>&1 || true
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
TRACK2_DEV_SEED_ROOT="$DEV" \
TRACK2_DEV_OUTPUT_ROOT="$OUT" \
TRACK2_CANDIDATE_CHECKPOINT="$CKPT" \
TRACK2_CANDIDATE_VARIANT="$VARIANT" \
TRACK2_DEV_BATCH_FILTER=00 \
TRACK2_SKIP_BASELINE=true \
  bash "$ROOT/pipeline/scripts/run_strict_track2_dev_eval.sh" >"$DRIVER_LOG" 2>&1

"$PY" - "$OUT/$VARIANT/batch_00/launcher.log" "$PREREG" "$REPORT" <<'PY'
import ast, json, pathlib, re, sys

log_path, prereg_path, report_path = map(pathlib.Path, sys.argv[1:])
payload = None
for line in reversed(log_path.read_text(encoding="utf-8", errors="replace").splitlines()):
    if "'eval/num_trajectories': 16" not in line:
        continue
    text = line[line.find("{"):]
    text = re.sub(r"array\(([^,\)]+)(?:, dtype=[^\)]+)?\)", r"\1", text)
    payload = ast.literal_eval(text)
    break
if payload is None:
    raise SystemExit("No complete 16-trajectory metric payload found")

def count(key):
    return int(round(float(payload[key]) * 16))

metrics = {
    "total": 16,
    "total_success": count("eval/success_once"),
    "left_success": count("eval/left_success"),
    "right_success": count("eval/right_success"),
    "grasp_once": count("eval/grasp_once"),
    "left_grasp": count("eval/left_grasp"),
    "right_grasp": count("eval/right_grasp"),
}
thresholds = json.loads(prereg_path.read_text())["acceptance"]
checks = {
    "total_success": metrics["total_success"] >= thresholds["total_success_at_least"],
    "left_success": metrics["left_success"] >= thresholds["left_success_at_least"],
    "right_success": metrics["right_success"] >= thresholds["right_success_at_least"],
    "grasp_once": metrics["grasp_once"] >= thresholds["grasp_once_at_least"],
}
result = {
    "format": "strict-track2-v412-public-batch00-result-v1",
    "variant": "v412_v411_oneupdate_public_batch00",
    "metrics": metrics,
    "acceptance_checks": checks,
    "accepted": all(checks.values()),
    "reserved_final128_used": False,
    "real_submission": False,
}
report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
PY
