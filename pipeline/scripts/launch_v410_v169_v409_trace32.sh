#!/usr/bin/env bash
set -euo pipefail

ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
O="$ROOT/artifacts/strict_track2_official_20260810"
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
P="$ROOT/pipeline/scripts"
RLINF="$ROOT/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
OPENPI="$ROOT/third_party/openpi-rlinf-full"
DIFFSYNTH="$O/official_deps/diffsynth_2a2e05f"
ADAPTER="$O/v15_frozen_input_adapter"
OPENPI_CKPT="$ROOT/artifacts/official_resources/pi05_adjust_bottle"
REWARD="$ROOT/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt"
T5="$ROOT/artifacts/official_resources/reward_model/t5-base"
NAME='v410_v169_v409_trace32_seed1568_20260823'
REG="$O/run_registry/$NAME"
RUN="$O/runs/$NAME"
V169="$O/runs/v169_parent_step5_seed1243_20260813/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'

test ! -e "$REG"
test ! -e "$RUN"
test -s "$J/v409_half_contracted_progressive_seed1567_20260823/audit/recursive_reward_causal.json"
mkdir -p "$REG" "$RUN/audit"

"$PY" - "$ROOT" "$REG/preregistration.json" <<'PY'
import datetime, hashlib, json, pathlib, sys
root, output = map(pathlib.Path, sys.argv[1:])
j = root / "artifacts/strict_track2_joint_augmentation_20260810"
paths = {
    "v409_runtime": root / "pipeline/wam_pipeline/v409_half_contracted_progressive_runtime.py",
    "trace_runtime": root / "pipeline/wam_pipeline/v410_v409_route_trace_runtime.py",
    "backend": root / "pipeline/wam_pipeline/backends.py",
    "service": root / "pipeline/scripts/restart_v410_v409_route_trace_services.sh",
    "launcher": root / "pipeline/scripts/launch_v410_v169_v409_trace32.sh",
    "v409_audit": j / "v409_half_contracted_progressive_seed1567_20260823/audit/recursive_reward_causal.json",
}
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
audit = json.loads(paths["v409_audit"].read_text())
if not audit.get("passed"):
    raise RuntimeError("v409 recursive gate did not pass")
payload = {
    "format": "strict-track2-v410-v169-v409-output-changing-trace32-preregistration-v1",
    "registered_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "purpose": "Verify that frozen v409 supplies reachable, within-group rankable right-arm progression before any policy update.",
    "world_model": "v410 telemetry wrapper is RGB-output-equivalent to frozen v409",
    "protocol": {"trajectories": 32, "requests": 800, "group_size": 4, "policy_updates": 0, "checkpoint_writes": 0, "actor_seed": 1568},
    "decision": {
        "progressive_routes_min": 48,
        "contracted_alpha_ge_0p125_min": 48,
        "groups_with_progressive_min": 20,
        "groups_with_alpha_spread_ge_0p05_min": 12,
        "all_progressive_advance_exactly_8": True,
        "all_contracted_alpha_le_0p5": True,
        "all_required": True,
        "on_fail": "Do not update policy; diagnose v409-policy interaction.",
        "on_pass": "Authorize one conservative RL update initialized from frozen v169.",
    },
    "evidence_sha256": {name: sha(path) for name, path in paths.items()},
    "guards": {"policy_modified": False, "outcomes_used_for_fit_or_selection": False, "hidden_or_final_data": False, "real_submission": False},
}
output.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY

cleanup() {
  "$PY" -m ray.scripts.scripts stop --force >/dev/null 2>&1 || true
  TRACK2_SERVICE_REG="$REG" bash "$P/restart_v410_v409_route_trace_services.sh" stop >/dev/null 2>&1 || true
  bash "$P/restart_v218_services.sh" start >"$REG/restore_v218.log" 2>&1 || true
}
trap cleanup EXIT

TRACK2_SERVICE_REG="$REG" TRACK2_CPUSET=0-5 \
  bash "$P/restart_v410_v409_route_trace_services.sh" start \
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
  runner.logger.log_path="$RUN" runner.ckpt_path="$V169" actor.seed=1568 \
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
progressive = [row for row in rows if row["progressive_route"]]
alpha = np.asarray([row["contracted_alpha"] for row in progressive], dtype=float)
advance = np.asarray([row["progressive_target_advance"] for row in progressive], dtype=int)
groups = []
for batch in lines:
    if len(batch) != 8:
        raise RuntimeError(f"expected native microbatch 8, got {len(batch)}")
    for begin in (0, 4):
        group = batch[begin:begin + 4]
        values = np.asarray([row["contracted_alpha"] for row in group], dtype=float)
        candidate = any(row["progressive_route"] for row in group)
        groups.append({"candidate": candidate, "spread": float(values.max() - values.min()), "values": values.tolist()})
candidate_groups = [group for group in groups if group["candidate"]]
spread_groups = [group for group in candidate_groups if group["spread"] >= .05]
gate = prereg["decision"]
checks = {
    "exact_requests": len(rows) == 800,
    "zero_policy_updates": raw["protocol"]["policy_updates"] == 0,
    "zero_checkpoint_writes": raw["protocol"].get("checkpoint_writes", 0) == 0,
    "progressive_routes": len(progressive) >= gate["progressive_routes_min"],
    "contracted_alpha_ge_0p125": int((alpha >= .125).sum()) >= gate["contracted_alpha_ge_0p125_min"],
    "groups_with_progressive": len(candidate_groups) >= gate["groups_with_progressive_min"],
    "groups_with_alpha_spread_ge_0p05": len(spread_groups) >= gate["groups_with_alpha_spread_ge_0p05_min"],
    "progressive_advance_exactly_8": bool(alpha.size and np.all(advance == 8)),
    "contracted_alpha_le_0p5": bool(alpha.size and np.all(alpha <= .5000001)),
}
q = (0, .1, .25, .5, .75, .9, 1)
quantiles = lambda values: {str(value): float(np.quantile(values, value)) for value in q} if len(values) else {}
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
result = {
    "format": "strict-track2-v410-v169-v409-output-changing-trace32-result-v1",
    "passed": all(checks.values()),
    "requests": len(rows),
    "progressive_routes": len(progressive),
    "terminal_precedence_routes": sum(row["terminal_precedence"] for row in rows),
    "contracted_alpha_ge_0p125": int((alpha >= .125).sum()),
    "groups_total": len(groups),
    "groups_with_progressive": len(candidate_groups),
    "groups_with_alpha_spread_ge_0p05": len(spread_groups),
    "contracted_alpha_quantiles": quantiles(alpha),
    "progressive_advance_quantiles": quantiles(advance),
    "checks": checks,
    "rollout_metrics_context_only_not_selection": raw["metrics"],
    "evidence_sha256": {"preregistration": sha(prereg_path), "raw": sha(raw_path), "trace": sha(trace_path)},
    "guards": {"policy_updates": 0, "checkpoint_writes": 0, "v409_rgb_output_equivalent": True, "hidden_or_final_data": False, "real_submission": False},
}
output_path.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY
