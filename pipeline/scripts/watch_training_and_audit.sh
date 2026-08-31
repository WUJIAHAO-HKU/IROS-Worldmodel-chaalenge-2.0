#!/usr/bin/env bash
set -euo pipefail

name=${1:?run name required}
training_screen=${2:?training screen required}
base='/root/autodl-tmp/IROS_WAM_2.0 challenge'
off="$base/artifacts/strict_track2_official_20260810"
run="$off/runs/$name"
prereg="$off/run_registry/$name/preregistration.json"
py='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
audit="$run/audit/p3_training_acceptance.json"
status="$run/audit/post_training_watcher_status.json"

while screen -ls 2>/dev/null | grep -q "[.]$training_screen"; do sleep 60; done
checkpoint="$run/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_8/actor/model_state_dict/full_weights.pt"
if [[ ! -s "$checkpoint" ]]; then
  "$py" - "$status" "$run/launcher.log" <<'PY'
import json, sys
from pathlib import Path
status, log = map(Path, sys.argv[1:])
status.write_text(json.dumps({"passed": False, "checkpoint_present": False,
    "launcher_tail": log.read_text(errors="replace")[-12000:]}, indent=2) + "\n")
PY
  exit 3
fi
set +e
"$py" "$base/pipeline/scripts/audit_strict_track2_kl_smoke.py" \
  --run "$run" --preregistration "$prereg" --output "$audit" \
  > "$run/audit/p3_training_acceptance.log" 2>&1
rc=$?
set -e
"$py" - "$status" "$audit" "$rc" <<'PY'
import json, sys
from pathlib import Path
status, audit, rc = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
status.write_text(json.dumps({"passed": rc == 0, "checkpoint_present": True,
    "audit": str(audit), "audit_exit_code": rc}, indent=2) + "\n")
PY
exit "$rc"
