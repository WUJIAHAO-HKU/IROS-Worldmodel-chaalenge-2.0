#!/usr/bin/env bash
set -euo pipefail

ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
O="$ROOT/artifacts/strict_track2_official_20260810"
P="$ROOT/pipeline/scripts"
RLINF="$ROOT/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
OPENPI="$ROOT/third_party/openpi-rlinf-full"
DIFFSYNTH="$O/official_deps/diffsynth_2a2e05f"
ADAPTER="$O/v15_frozen_input_adapter"
OPENPI_CKPT="$ROOT/artifacts/official_resources/pi05_adjust_bottle"
REWARD="$ROOT/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt"
T5="$ROOT/artifacts/official_resources/reward_model/t5-base"
NAME='v405_v169_v400_progressive_trace32_seed1563_20260823'
REG="$O/run_registry/$NAME"
RUN="$O/runs/$NAME"
V169="$O/runs/v169_parent_step5_seed1243_20260813/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'

test ! -e "$REG"
test ! -e "$RUN"
mkdir -p "$REG" "$RUN/audit"

"$PY" - "$ROOT" "$REG/preregistration.json" <<'PY'
import datetime, hashlib, json, pathlib, sys
root, output = map(pathlib.Path, sys.argv[1:])
paths = {
    "runtime": root / "pipeline/wam_pipeline/v405_v400_progressive_diagnostic_runtime.py",
    "base_runtime": root / "pipeline/wam_pipeline/v404_v400_dense_diagnostic_runtime.py",
    "backend": root / "pipeline/wam_pipeline/backends.py",
    "service": root / "pipeline/scripts/restart_v405_progressive_diagnostic_services.sh",
    "launcher": root / "pipeline/scripts/launch_v405_v169_v400_progressive_trace32.sh",
}
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
payload = {
    "format": "strict-track2-v405-progressive-trace32-preregistration-v1",
    "registered_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "purpose": "Test whether public-expert progressive successors provide reachable within-GRPO-group action ranking before changing RGB or training policy.",
    "output_equivalence": "exact v400 RGB; telemetry only",
    "protocol": {"trajectories": 32, "requests": 800, "group_size": 4, "policy_updates": 0, "checkpoint_writes": 0, "actor_seed": 1563},
    "decision": {
        "broad_eligible_min": 96,
        "progressive_alpha_ge_0p25_min": 48,
        "positive_alignment_min": 64,
        "progressive_advance_positive_min": 96,
        "groups_with_candidate_min": 20,
        "groups_with_alpha_spread_ge_0p10_min": 12,
        "all_required": True,
        "on_fail": "Reject progressive-successor integration without RL.",
        "on_pass": "Authorize one frozen progressive integration and public holdout recursive reward audit; still no policy update until audit passes.",
    },
    "evidence_sha256": {name: sha(path) for name, path in paths.items()},
    "guards": {"world_model_rgb_modified_vs_v400": False, "policy_modified": False, "public_expert_library_only": True, "outcomes_used_for_fit": False, "hidden_or_final_data": False, "real_submission": False},
}
output.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY

cleanup() {
  "$PY" -m ray.scripts.scripts stop --force >/dev/null 2>&1 || true
  TRACK2_SERVICE_REG="$REG" bash "$P/restart_v405_progressive_diagnostic_services.sh" stop >/dev/null 2>&1 || true
  bash "$P/restart_v218_services.sh" start >"$REG/restore_v218.log" 2>&1 || true
}
trap cleanup EXIT

TRACK2_SERVICE_REG="$REG" TRACK2_CPUSET=0-5 \
  bash "$P/restart_v405_progressive_diagnostic_services.sh" start \
  >"$REG/restart_trace_before_rollout.log" 2>&1
"$PY" -m ray.scripts.scripts stop --force >"$RUN/ray_stop_before.log" 2>&1 || true
taskset -c 6-21 "$PY" -m ray.scripts.scripts start --head \
  --object-store-memory=4294967296 --disable-usage-stats >"$RUN/ray_start.log" 2>&1

cd "$RLINF"
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 RAYON_NUM_THREADS=2
export TRACK2_ROLLOUT_ONLY_OUTPUT="$RUN/audit/raw.json"
REPO_PATH="$RLINF" EMBODIED_PATH="$RLINF/examples/embodiment" \
OPENPI_CKPT_PATH="$OPENPI_CKPT" WAN_ROBOTWIN_ADJUST_BOTTLE_CKPT="$ADAPTER" \
ROBOTWIN_REWARD_MODEL_PATH="$REWARD" T5_MODEL_PATH="$T5" \
PYTHONPATH="$DIFFSYNTH:$OPENPI/packages/openpi-client/src:$OPENPI/src:$RLINF" \
PYTHONHASHSEED=0 RAY_DEDUP_LOGS=0 taskset -c 6-21 "$PY" \
  "$P/trainmode_rollout_only_embodied_agent.py" \
  --config-path "$RLINF/examples/embodiment/config" \
  --config-name wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05 \
  runner.logger.log_path="$RUN" runner.ckpt_path="$V169" actor.seed=1563 \
  algorithm.rollout_epoch=1 algorithm.group_size=4 env.train.total_num_envs=32 \
  env.train.group_size=4 env.train.seed=0 env.train.max_episode_steps=200 \
  env.train.max_steps_per_rollout_epoch=200 \
  env.train.http.server_url=http://127.0.0.1:18084 \
  env.train.wan_wm_hf_ckpt_path="$ADAPTER" \
  env.train.VAE_path="$ADAPTER/Wan2.2_VAE.pth" \
  env.train.model_path="$ADAPTER/dit_model.safetensors" \
  env.train.initial_image_path="$ADAPTER/dataset/" env.train.enable_kir=false \
  env.train.video_cfg.video_base_dir="$RUN/video/train" \
  env.train.enable_offload=false rollout.enable_offload=false \
  >"$RUN/launcher.log" 2>&1

cd "$ROOT"
cp "$REG/route_trace.jsonl" "$RUN/audit/route_trace.jsonl"
"$PY" - "$REG/preregistration.json" "$RUN/audit/raw.json" \
  "$RUN/audit/route_trace.jsonl" "$REG/result.json" <<'PY'
import hashlib, json, pathlib, sys
import numpy as np

prereg_path, raw_path, trace_path, output_path = map(pathlib.Path, sys.argv[1:])
prereg = json.loads(prereg_path.read_text())
raw = json.loads(raw_path.read_text())
lines = [json.loads(line)["batch"] for line in trace_path.read_text().splitlines()]
rows = [item for batch in lines for item in batch]
broad = [row for row in rows if row["broad_eligible"]]
alpha = np.asarray([row["progressive_alpha"] for row in broad], dtype=float)
distance = np.asarray([row["progressive_action_distance"] for row in broad], dtype=float)
alignment = np.asarray([row["progressive_alignment"] for row in broad], dtype=float)
advance = np.asarray([row["progressive_target_advance"] for row in broad], dtype=float)
groups = []
for batch in lines:
    if len(batch) != 8:
        raise RuntimeError(f"expected native microbatch 8, got {len(batch)}")
    for begin in (0, 4):
        group = batch[begin:begin + 4]
        values = np.asarray([row["progressive_alpha"] or 0.0 for row in group])
        candidate = any(row["broad_eligible"] for row in group)
        groups.append({"candidate": candidate, "spread": float(values.max() - values.min()), "values": values.tolist()})
candidate_groups = [group for group in groups if group["candidate"]]
spread_groups = [group for group in candidate_groups if group["spread"] >= .10]
q = (0, .1, .25, .5, .75, .9, .95, .99, 1)
quantiles = lambda values: {str(value): float(np.quantile(values, value)) for value in q}
gate = prereg["decision"]
checks = {
    "exact_requests": len(rows) == 800,
    "zero_policy_updates": raw["protocol"]["policy_updates"] == 0,
    "broad_eligible": len(broad) >= gate["broad_eligible_min"],
    "progressive_alpha_ge_0p25": int((alpha >= .25).sum()) >= gate["progressive_alpha_ge_0p25_min"],
    "positive_alignment": int((alignment > 0).sum()) >= gate["positive_alignment_min"],
    "progressive_advance_positive": int((advance > 0).sum()) >= gate["progressive_advance_positive_min"],
    "groups_with_candidate": len(candidate_groups) >= gate["groups_with_candidate_min"],
    "groups_with_alpha_spread_ge_0p10": len(spread_groups) >= gate["groups_with_alpha_spread_ge_0p10_min"],
}
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
result = {
    "format": "strict-track2-v405-progressive-trace32-result-v1",
    "passed": all(checks.values()),
    "requests": len(rows),
    "broad_eligible": len(broad),
    "progressive_alpha_ge_0p25": int((alpha >= .25).sum()),
    "positive_alignment": int((alignment > 0).sum()),
    "progressive_advance_positive": int((advance > 0).sum()),
    "groups_total": len(groups),
    "groups_with_candidate": len(candidate_groups),
    "groups_with_alpha_spread_ge_0p10": len(spread_groups),
    "progressive_alpha_quantiles": quantiles(alpha),
    "progressive_action_distance_quantiles": quantiles(distance),
    "progressive_alignment_quantiles": quantiles(alignment),
    "progressive_advance_quantiles": quantiles(advance),
    "checks": checks,
    "rollout_metrics_context_only_not_selection": raw["metrics"],
    "evidence_sha256": {"preregistration": sha(prereg_path), "raw": sha(raw_path), "trace": sha(trace_path)},
    "guards": {"policy_updates": 0, "checkpoint_writes": 0, "v400_rgb_output_equivalent": True, "hidden_or_final_data": False, "real_submission": False},
}
output_path.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY
