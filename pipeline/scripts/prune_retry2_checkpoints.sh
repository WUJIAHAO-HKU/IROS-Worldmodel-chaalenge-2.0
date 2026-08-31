#!/usr/bin/env bash
set -euo pipefail
ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
RUN="$ROOT/artifacts/strict_track2_official_20260810/runs/probability_consistent_four_step_v169_seed1243_retry2"
CKPT="$RUN/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints"
ACTIVE="$RUN/audit/training_active"
LOG="$RUN/audit/checkpoint_retention.log"
touch "$ACTIVE"
while [[ -e "$ACTIVE" ]]; do
  mapfile -t complete < <(find "$CKPT" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' -print | while read -r d; do
    [[ -s "$d/actor/dcp_checkpoint/.metadata" && -s "$d/actor/model_state_dict/full_weights.pt" ]] && echo "$d"
  done | sort -V)
  n=${#complete[@]}
  if (( n > 2 )); then
    for ((i=0; i<n-2; i++)); do
      d="${complete[$i]}"; case "$d" in "$CKPT"/global_step_[0-9]*)
        printf '%s prune_complete=%s keep_latest=2\n' "$(date -Iseconds)" "$d" >> "$LOG"
        rm -rf -- "$d";; *) exit 9;; esac
    done
  fi
  sleep 30
done
