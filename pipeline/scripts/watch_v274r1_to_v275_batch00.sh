#!/usr/bin/env bash
set -euo pipefail
ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
O="$ROOT/artifacts/strict_track2_official_20260810"
RL_NAME='v274_v271_fresh_fullbudget_h200_r8_step5_lr2e5_beta001_seed1471_retry1_20260819'
RUN="$O/runs/$RL_NAME"
for _ in $(seq 1 2880); do
  if [[ -f "$RUN/audit/V274_RETRY1_TRAINING_ACCEPTED" ]]; then
    test -f "$RUN/audit/p3_training_acceptance.json"
    test ! -e "$RUN/audit/PUBLIC_BATCH00_PASSED"
    test ! -e "$RUN/audit/PUBLIC_BATCH00_REJECTED"
    # The training launcher writes the marker before its EXIT trap finishes.
    # Wait until that trap (including legacy service restoration) is complete,
    # so the batch00 launcher can establish one deterministic GPU environment.
    while screen -ls 2>/dev/null | grep -q '[.]v274r1_v271_rl'; do sleep 5; done
    screen -ls | grep -q '[.]v275_v274r1_b00' && exit 9 || true
    screen -dmS v275_v274r1_b00 bash -lc "bash \"$ROOT/pipeline/scripts/launch_v275_v274_batch00_gate.sh\" >\"$O/run_registry/v275_v274r1_batch00_launcher.log\" 2>&1"
    echo V275_V274R1_BATCH00_LAUNCHED
    exit 0
  fi
  if ! screen -ls | grep -q '[.]v274r1_v271_rl'; then
    echo V274_RETRY1_STOPPED_WITHOUT_TRAINING_ACCEPTANCE >&2
    exit 3
  fi
  sleep 30
done
echo V274_RETRY1_WATCH_TIMEOUT >&2
exit 4
