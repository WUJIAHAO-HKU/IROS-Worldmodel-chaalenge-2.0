#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-/root/autodl-tmp/IROS_WAM_2.0 challenge}
ACTOR_SEED=${2:-1240}
case "$ACTOR_SEED" in 1240|1235|1237) ;; *)
  printf 'Actor seed %s is outside the preregistered V15.6 order\n' "$ACTOR_SEED" >&2
  exit 2
esac

AUDIT="$ROOT/artifacts/strict_track2_official_20260810"
JOINT="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
RUN_LABEL="${TRACK2_RUN_LABEL:-official_replica_v156_absstate_seed${ACTOR_SEED}}"
VARIANT="${TRACK2_CANDIDATE_VARIANT:-v156_absstate_seed${ACTOR_SEED}_historical_dev22}"
RUN="$AUDIT/runs/$RUN_LABEL"
EXPERIMENT="wan_robotwin_adjust_bottle_http_grpo_openpi_pi05"
ACTOR="$RUN/$EXPERIMENT/checkpoints/global_step_1/actor"
CHECKPOINT="$ACTOR/model_state_dict/full_weights.pt"
BRIDGE_AUDIT="${TRACK2_BRIDGE_AUDIT:-$JOINT/v156_bridge_audit_absstate_seed${ACTOR_SEED}}"
ROLLOUT="$BRIDGE_AUDIT/rollout_000000.npz"
REPORT="$BRIDGE_AUDIT/report.json"
ALIGNMENT_REPORT="$BRIDGE_AUDIT/initial_alignment_report.json"
REFERENCE_WINDOW="${TRACK2_REFERENCE_WINDOW:-$ROOT/artifacts/adjust_bottle_windows_full/episode49_00000.npz}"
REQUIRE_INITIAL_ALIGNMENT="${TRACK2_REQUIRE_INITIAL_ALIGNMENT:-0}"
RL_PY=/root/autodl-tmp/conda_envs/rlinf_track2/bin/python

test -s "$CHECKPOINT"
grep -aq 'Global Step:    1/1' "$RUN/launcher.log"
mapfile -t rollout_files < <(find "$BRIDGE_AUDIT" -maxdepth 1 -type f -name 'rollout_*.npz' -print | sort)
if [[ "${#rollout_files[@]}" -ne 1 || "${rollout_files[0]}" != "$ROLLOUT" ]]; then
  printf 'Expected exactly one frozen official rollout archive, found %s\n' "${#rollout_files[@]}" >&2
  exit 3
fi

if [[ "$REQUIRE_INITIAL_ALIGNMENT" == "1" ]]; then
  PYTHONPATH="$ROOT/pipeline" "$RL_PY" \
    "$ROOT/pipeline/scripts/audit_strict_track2_initial_alignment.py" \
    --audit "$ROLLOUT" \
    --reference-window "$REFERENCE_WINDOW" \
    --output "$ALIGNMENT_REPORT"
  cp "$ALIGNMENT_REPORT" "$RUN/audit/initial_alignment_report.json"
elif [[ "$REQUIRE_INITIAL_ALIGNMENT" != "0" ]]; then
  printf 'TRACK2_REQUIRE_INITIAL_ALIGNMENT must be 0 or 1\n' >&2
  exit 4
fi

if [[ ! -s "$REPORT" ]]; then
  PYTHONPATH="$ROOT/pipeline" "$RL_PY" \
    "$ROOT/pipeline/scripts/audit_strict_track2_bridge_rollout.py" \
    --audit "$ROLLOUT" \
    --release "$JOINT/v156_action_gated_reward_safe_blend12_formal_release" \
    --library "$ROOT/artifacts" \
    --output "$REPORT" \
    --device cpu \
    --expected-route candidate
fi

"$RL_PY" - <<'PY' "$REPORT"
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
assert report["batch_size"] == 2, report
assert report["candidate_routes"] == 2, report
assert report["all_replays_bit_exact"] is True, report
assert report["group_future_action_l2"] > 0.0, report
assert report["future_action_max"] > 0.5, report
assert report["future_gripper_mean"]["left"] > 0.5, report
assert report["future_gripper_mean"]["right"] > 0.5, report
assert report["participant_action_selection"] is False, report
assert report["mpc"] is False, report
PY
cp "$REPORT" "$RUN/audit/bridge_rollout_absolute_action_report.json"

# The consolidated full_weights file is the formal policy checkpoint.  DCP is
# an exact redundant serialization and is removed only after its hashes were
# written by the official runner.
DCP="$ACTOR/dcp_checkpoint"
if [[ -d "$DCP" ]]; then
  [[ ! -L "$DCP" ]]
  bytes=$(du -sb "$DCP" | cut -f1)
  rm -rf -- "$DCP"
  [[ ! -e "$DCP" ]]
  printf 'DELETED_EXACT=%s\nBYTES=%s\nRECOVERABLE=false\n' "$DCP" "$bytes" \
    >"$RUN/audit/dcp_cleanup_record.txt"
fi

TRACK2_CANDIDATE_CHECKPOINT="$CHECKPOINT" \
TRACK2_CANDIDATE_VARIANT="$VARIANT" \
TRACK2_DEV_SEED_ROOT="$AUDIT/real_robotwin_eval/development_seeds22" \
TRACK2_DEV_OUTPUT_ROOT="$AUDIT/real_robotwin_eval/development_metrics" \
TRACK2_DEV_BATCH_FILTER="00" \
  "$AUDIT/real_robotwin_eval/run_strict_track2_dev_eval.sh"

PYTHONPATH="$ROOT/pipeline/scripts" "$RL_PY" \
  "$ROOT/pipeline/scripts/screen_strict_track2_dev22_batch00.py" \
  --output-root "$AUDIT/real_robotwin_eval/development_metrics" \
  --dev-root "$AUDIT/real_robotwin_eval/development_seeds22" \
  --candidate "$VARIANT" \
  --output "$AUDIT/real_robotwin_eval/dev_batch00_screen_$VARIANT.json"

TRACK2_CANDIDATE_CHECKPOINT="$CHECKPOINT" \
TRACK2_CANDIDATE_VARIANT="$VARIANT" \
TRACK2_DEV_SEED_ROOT="$AUDIT/real_robotwin_eval/development_seeds22" \
TRACK2_DEV_OUTPUT_ROOT="$AUDIT/real_robotwin_eval/development_metrics" \
TRACK2_DEV_BATCH_FILTER="01" \
  "$AUDIT/real_robotwin_eval/run_strict_track2_dev_eval.sh"

PYTHONPATH="$ROOT/pipeline/scripts" "$RL_PY" \
  "$ROOT/pipeline/scripts/summarize_strict_track2_dev_eval.py" \
  --output-root "$AUDIT/real_robotwin_eval/development_metrics" \
  --dev-root "$AUDIT/real_robotwin_eval/development_seeds22" \
  --candidate "$VARIANT" \
  --output "$AUDIT/real_robotwin_eval/dev_ranking_$VARIANT.json"

printf 'ABSSTATE_POLICY_PRESCREEN_COMPLETE variant=%s checkpoint=%s\n' \
  "$VARIANT" "$CHECKPOINT"
