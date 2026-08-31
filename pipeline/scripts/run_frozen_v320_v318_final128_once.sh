#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$BASE/artifacts/strict_track2_official_20260810"
EVAL="$OFF/real_robotwin_eval"
P="$BASE/pipeline/scripts"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
VARIANT='v319_v318_v317_rtx5090_step10_seed1471'
FROZEN="$OFF/frozen_candidates/v320_v318_v317_rtx5090_step10_seed1471"
FREEZE="$FROZEN/freeze_manifest.json"
PREREG="$FROZEN/final128_preregistration.json"
SEEDS="$EVAL/final128_effective_seed_bundle_manifest.json"
OUTPUT_ROOT="$EVAL/frozen_v320_v318_v317_rtx5090_step10_seed1471_final128"
RESULT="$FROZEN/final128_result.json"
LOCK="$FROZEN/final128_execution.lock"
INVOCATION="$FROZEN/final128_execution.json"
RUNNER="$P/run_strict_track2_final128_eval.sh"
SUMMARIZER="$P/summarize_frozen_final128.py"
WRAPPER="$P/run_frozen_v320_v318_final128_once.sh"
CPUSET='0-21'

for path in "$FREEZE" "$PREREG" "$SEEDS" "$RUNNER" "$SUMMARIZER" "$WRAPPER"; do
  test -s "$path"
done
test ! -e "$FROZEN/FINAL128_EVALUATION_COMPLETE"

"$PY" - "$FREEZE" "$PREREG" "$SEEDS" "$RUNNER" "$SUMMARIZER" "$WRAPPER" "$OUTPUT_ROOT" "$VARIANT" <<'PY'
import hashlib,json,sys
from pathlib import Path
freeze_path,prereg_path,seeds_path,runner,summarizer,wrapper,output_root=map(Path,sys.argv[1:8])
variant=sys.argv[8]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''): h.update(block)
 return h.hexdigest()
freeze=json.loads(freeze_path.read_text()); prereg=json.loads(prereg_path.read_text()); seeds=json.loads(seeds_path.read_text())
assert freeze['unique_final_candidate'] is True and freeze['official_submission'] is False
assert freeze['post_final_selection_allowed'] is False and freeze['variant']==variant
assert sha(Path(freeze['checkpoint']))==freeze['checkpoint_sha256']
assert prereg['authorized'] is True and prereg['contest_submission'] is False
assert prereg['selection_after_this_evaluation'] is False and prereg['variant']==variant
assert prereg['checkpoint_sha256']==freeze['checkpoint_sha256']
assert prereg['freeze_manifest_sha256']==sha(freeze_path)
assert prereg['final_seed_manifest_sha256']==sha(seeds_path)
assert prereg['output_root']==str(output_root) and prereg['target']=={'count':128,'successes_min':85}
assert seeds['result_data_used'] is False and seeds['seed_count']==128 and seeds['unique_seed_count']==128
evidence=freeze['evidence']
assert evidence['local_final128_runner']['sha256']==sha(runner)
assert evidence['local_final128_summarizer']['sha256']==sha(summarizer)
assert evidence['local_final128_one_candidate_wrapper']['sha256']==sha(wrapper)
print('V320_FROZEN_FINAL128_INPUTS_VERIFIED')
PY

if [[ ! -d "$LOCK" && -e "$OUTPUT_ROOT" ]]; then
  echo 'refusing first invocation because final output root already exists' >&2
  exit 5
fi
if mkdir "$LOCK" 2>/dev/null; then
  "$PY" - "$FREEZE" "$PREREG" "$SEEDS" "$INVOCATION" "$VARIANT" <<'PY'
import hashlib,json,os,sys
from datetime import datetime,timezone
from pathlib import Path
freeze,prereg,seeds,output=map(Path,sys.argv[1:5]); variant=sys.argv[5]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
payload={'format':'strict-track2-v320-frozen-final128-execution-v1','started_utc':datetime.now(timezone.utc).isoformat(),'contest_submission':False,'variant':variant,'freeze_manifest_sha256':sha(freeze),'preregistration_sha256':sha(prereg),'seed_bundle_sha256':sha(seeds),'resume_policy':'only incomplete batches of this exact frozen candidate'}
tmp=output.with_suffix('.tmp'); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); os.replace(tmp,output)
PY
else
  test -s "$INVOCATION"
fi

CHECKPOINT=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$PREREG")
restore() {
  TRACK2_CPUSET=0-7 bash "$P/restart_v271_v274_services.sh" start >"$FROZEN/restart_v271_after_final128.log" 2>&1 || true
}
trap restore EXIT
bash "$P/restart_v271_v274_services.sh" stop || true
bash "$P/restart_v317_services.sh" stop || true
"$PY" -m ray.scripts.scripts stop --force >"$FROZEN/ray_stop_before_final128.log" 2>&1 || true
TRACK2_INSTRUMENTED_OUTPUT_ROOT="$OUTPUT_ROOT" \
TRACK2_FINAL128_CHECKPOINT="$CHECKPOINT" \
TRACK2_FINAL128_VARIANT="$VARIANT" \
TRACK2_FINAL128_RUN_BASELINE=false \
  taskset -c "$CPUSET" bash "$RUNNER" >"$FROZEN/final128_runner.log" 2>&1

set +e
"$PY" "$SUMMARIZER" --preregistration "$PREREG" --seed-root "$EVAL" --output-root "$OUTPUT_ROOT" --output "$RESULT" >"$FROZEN/final128_summary.log" 2>&1
rc=$?
set -e
if (( rc == 0 )); then
  touch "$FROZEN/FINAL128_TARGET_REACHED"
elif (( rc == 2 )); then
  touch "$FROZEN/FINAL128_TARGET_NOT_REACHED"
else
  exit "$rc"
fi
touch "$FROZEN/FINAL128_EVALUATION_COMPLETE"
exit "$rc"
