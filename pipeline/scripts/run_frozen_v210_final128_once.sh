#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$BASE/artifacts/strict_track2_official_20260810"
EVAL="$OFF/real_robotwin_eval"
FROZEN="$OFF/frozen_candidates/v210_step8_seed1408"
FREEZE="$FROZEN/freeze_manifest.json"
PREREG="$FROZEN/final128_preregistration.json"
SEED_BUNDLE="$EVAL/final128_effective_seed_bundle_manifest.json"
OUTPUT_ROOT="$EVAL/frozen_v210_step8_seed1408_final128"
RESULT="$FROZEN/final128_result.json"
LOCK="$FROZEN/final128_execution.lock"
INVOCATION="$FROZEN/final128_execution.json"
RUNNER="$BASE/pipeline/scripts/run_strict_track2_final128_eval.sh"
SUMMARIZER="$BASE/pipeline/scripts/summarize_frozen_final128.py"
WRAPPER="$BASE/pipeline/scripts/run_frozen_v210_final128_once.sh"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'

for path in "$FREEZE" "$PREREG" "$SEED_BUNDLE" "$RUNNER" "$SUMMARIZER" "$WRAPPER"; do test -s "$path"; done
test ! -e "$FROZEN/FINAL128_EVALUATION_COMPLETE"

"$PY" - "$FREEZE" "$PREREG" "$SEED_BUNDLE" "$RUNNER" "$SUMMARIZER" "$WRAPPER" "$OUTPUT_ROOT" <<'PY'
import hashlib, json, sys
from pathlib import Path
freeze_path, prereg_path, seed_path, runner, summarizer, wrapper, output_root = map(Path, sys.argv[1:])
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
freeze=json.loads(freeze_path.read_text()); prereg=json.loads(prereg_path.read_text()); seeds=json.loads(seed_path.read_text())
assert freeze.get("unique_final_candidate") is True and freeze.get("official_submission") is False
assert freeze["variant"] == "v210_step8_seed1408"
assert digest(Path(freeze["checkpoint"])) == freeze["checkpoint_sha256"]
assert prereg.get("authorized") is True and prereg.get("contest_submission") is False
assert prereg.get("selection_after_this_evaluation") is False
assert prereg["variant"] == freeze["variant"] and prereg["checkpoint_sha256"] == freeze["checkpoint_sha256"]
assert prereg["freeze_manifest_sha256"] == digest(freeze_path)
assert prereg["final_seed_manifest"] == str(seed_path) and prereg["final_seed_manifest_sha256"] == digest(seed_path)
assert prereg["output_root"] == str(output_root) and prereg["target"] == {"count":128,"successes_min":85}
assert seeds["result_data_used"] is False and seeds["seed_count"] == 128 and seeds["unique_seed_count"] == 128
assert len(seeds["batches"]) == 9
for item in seeds["batches"]:
    path=Path(item["path"]); assert path.is_file() and digest(path) == item["sha256"]
evidence=freeze["evidence"]
assert evidence["local_final128_runner"]["sha256"] == digest(runner)
assert evidence["local_final128_summarizer"]["sha256"] == digest(summarizer)
assert evidence["local_final128_one_candidate_wrapper"]["sha256"] == digest(wrapper)
print("FROZEN_FINAL128_INPUTS_VERIFIED")
PY

if test ! -d "$LOCK" && test -e "$OUTPUT_ROOT"; then
  echo 'refusing first invocation because final output root already exists' >&2; exit 5
fi
if mkdir "$LOCK" 2>/dev/null; then
  "$PY" - "$FREEZE" "$PREREG" "$SEED_BUNDLE" "$INVOCATION" <<'PY'
import hashlib,json,os,sys
from datetime import datetime,timezone
from pathlib import Path
freeze,prereg,seeds,output=map(Path,sys.argv[1:])
digest=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
payload={"format":"strict-track2-frozen-final128-execution-v1","started_utc":datetime.now(timezone.utc).isoformat(),"contest_submission":False,"variant":"v210_step8_seed1408","freeze_manifest_sha256":digest(freeze),"preregistration_sha256":digest(prereg),"seed_bundle_sha256":digest(seeds),"resume_policy":"only incomplete batches of this exact frozen candidate"}
tmp=output.with_suffix(output.suffix+".tmp"); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); os.replace(tmp,output)
PY
else
  test -s "$INVOCATION"
fi

"$PY" - "$INVOCATION" "$FREEZE" "$PREREG" "$SEED_BUNDLE" <<'PY'
import hashlib,json,sys
from pathlib import Path
invocation_path,freeze,prereg,seeds=map(Path,sys.argv[1:]); x=json.loads(invocation_path.read_text())
digest=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert x["contest_submission"] is False and x["variant"] == "v210_step8_seed1408"
assert x["freeze_manifest_sha256"] == digest(freeze) and x["preregistration_sha256"] == digest(prereg) and x["seed_bundle_sha256"] == digest(seeds)
print("FINAL128_EXECUTION_IDENTITY_VERIFIED")
PY

CHECKPOINT="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$PREREG" checkpoint)"
VARIANT="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$PREREG" variant)"
restart_v209() { bash "$BASE/pipeline/scripts/restart_v209_services.sh" start > "$FROZEN/restart_v209_after_final128.log" 2>&1 || true; }
trap restart_v209 EXIT
for session in wm_v209_bridge wm_v209_gpu; do screen -S "$session" -X quit >/dev/null 2>&1 || true; done

TRACK2_INSTRUMENTED_OUTPUT_ROOT="$OUTPUT_ROOT" TRACK2_FINAL128_CHECKPOINT="$CHECKPOINT" \
TRACK2_FINAL128_VARIANT="$VARIANT" TRACK2_FINAL128_RUN_BASELINE=false \
  bash "$RUNNER" > "$FROZEN/final128_runner.log" 2>&1
set +e
"$PY" "$SUMMARIZER" --preregistration "$PREREG" --seed-root "$EVAL" \
  --output-root "$OUTPUT_ROOT" --output "$RESULT" > "$FROZEN/final128_summary.log" 2>&1
rc=$?
set -e
if (( rc == 0 )); then touch "$FROZEN/FINAL128_TARGET_REACHED"; elif (( rc == 2 )); then touch "$FROZEN/FINAL128_TARGET_NOT_REACHED"; else exit "$rc"; fi
touch "$FROZEN/FINAL128_EVALUATION_COMPLETE"
exit "$rc"
