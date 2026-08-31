#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$BASE/artifacts/strict_track2_official_20260810"
NAME='v203b_v202_conservativekl_h200_r2_step8_seed1403_20260818'
RUN="$OFF/runs/$NAME"
PREREG="$OFF/run_registry/$NAME/preregistration.json"
AUDIT="$RUN/audit/p3_training_acceptance.json"
STATUS="$RUN/audit/post_training_watcher_status.json"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'

while screen -ls 2>/dev/null | grep -q '[.]v203b_rl_train'; do
  sleep 60
done

checkpoint="$RUN/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_8/actor/model_state_dict/full_weights.pt"
if [[ -s "$checkpoint" ]]; then
  set +e
  "$PY" "$BASE/pipeline/scripts/audit_strict_track2_kl_smoke.py" \
    --run "$RUN" \
    --preregistration "$PREREG" \
    --output "$AUDIT" > "$RUN/audit/p3_training_acceptance.log" 2>&1
  audit_rc=$?
  set -e
  "$PY" - "$STATUS" "$audit_rc" "$AUDIT" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path

path, rc, audit = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
path.write_text(json.dumps({
    "format": "strict-track2-v203b-post-training-watcher-v1",
    "completed_utc": datetime.now(timezone.utc).isoformat(),
    "checkpoint_present": True,
    "audit_exit_code": rc,
    "audit": str(audit),
    "passed": rc == 0,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  exit "$audit_rc"
fi

"$PY" - "$STATUS" "$RUN/launcher.log" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path

status, log = Path(sys.argv[1]), Path(sys.argv[2])
tail = log.read_text(encoding="utf-8", errors="replace")[-12000:] if log.exists() else ""
status.write_text(json.dumps({
    "format": "strict-track2-v203b-post-training-watcher-v1",
    "completed_utc": datetime.now(timezone.utc).isoformat(),
    "checkpoint_present": False,
    "passed": False,
    "launcher_log": str(log),
    "launcher_tail": tail,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
exit 3
