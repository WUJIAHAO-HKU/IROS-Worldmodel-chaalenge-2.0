#!/usr/bin/env bash
set -euo pipefail

cd '/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF='artifacts/strict_track2_official_20260810'
NAME='v318_v317_authorized_official_h200_r4_step10_lr2e5_beta001_seed1471_20260822'
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"
CKPT="$RUN/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_3/actor/model_state_dict/full_weights.pt"
OPT="$RUN/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_3/actor/optimizer_recovery.pt"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'

for _ in $(seq 1 160); do
  if [[ -s "$CKPT" && -s "$OPT" ]]; then
    old_ckpt=$(stat -c '%s:%Y' "$CKPT")
    old_opt=$(stat -c '%s:%Y' "$OPT")
    sleep 30
    new_ckpt=$(stat -c '%s:%Y' "$CKPT")
    new_opt=$(stat -c '%s:%Y' "$OPT")
    if [[ "$old_ckpt" == "$new_ckpt" && "$old_opt" == "$new_opt" ]] \
      && "$PY" -c 'import sys,zipfile; raise SystemExit(0 if all(zipfile.is_zipfile(p) for p in sys.argv[1:]) else 1)' "$CKPT" "$OPT"; then
      outer=$(pgrep -f '^bash /root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/launch_v300_v295_official_rl.sh$' | head -n 1)
      runner=$(pgrep -P "$outer" -f 'run_strict_track2_conservative_kl.sh' | head -n 1)
      driver=$(pgrep -P "$runner" -f 'train_embodied_agent.py' | head -n 1)
      test -n "$outer" && test -n "$runner" && test -n "$driver"
      kill -STOP "$outer"
      kill -TERM "$driver" || true
      for _ in $(seq 1 12); do kill -0 "$driver" 2>/dev/null || break; sleep 5; done
      "$PY" -m ray.scripts.scripts stop --force >"$REG/ray_stop_for_step3_gate.log" 2>&1 || true
      bash pipeline/scripts/restart_v317_services.sh stop >"$REG/v317_stop_for_step3_gate.log" 2>&1 || true
      printf 'paused_at=%s outer_pid=%s runner_pid=%s driver_pid=%s checkpoint=%s optimizer=%s\n' \
        "$(date -u +%FT%TZ)" "$outer" "$runner" "$driver" "$new_ckpt" "$new_opt" \
        >"$REG/V318_PAUSED_AT_GLOBAL_STEP3"
      echo V318_PAUSED_AT_GLOBAL_STEP3
      cat "$REG/V318_PAUSED_AT_GLOBAL_STEP3"
      exit 0
    fi
  fi
  if ! screen -ls 2>/dev/null | grep -q '[.]v318_v317_official_rl'; then
    echo TRAINING_SCREEN_GONE
    exit 3
  fi
  sleep 30
done

echo GLOBAL_STEP3_WAIT_TIMEOUT
exit 5
