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
NAME='v404_v169_v400_dense_trace32_seed1562_20260823'
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
    "runtime": root / "pipeline/wam_pipeline/v404_v400_dense_diagnostic_runtime.py",
    "backend": root / "pipeline/wam_pipeline/backends.py",
    "service": root / "pipeline/scripts/restart_v404_dense_diagnostic_services.sh",
    "launcher": root / "pipeline/scripts/launch_v404_v169_v400_dense_trace32.sh",
    "v400_manifest": root / "artifacts/strict_track2_joint_augmentation_20260810/v400_supported_posterior_blend_seed1560_20260823/release/supported_posterior_blend_manifest.json",
}
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
payload = {
    "format": "strict-track2-v404-v400-dense-trace32-preregistration-v1",
    "registered_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "purpose": "Measure dense right post-grasp action/phase/endpoint coverage before authorizing another RL update.",
    "output_equivalence": "exact v400 RGB path; telemetry side effect only",
    "protocol": {"trajectories": 32, "rollout_epochs": 1, "episode_steps": 200, "policy_updates": 0, "checkpoint_writes": 0, "actor_seed": 1562},
    "decision": {
        "broad_eligible_min": 96,
        "phase_probability_ge_0p5_min": 12,
        "endpoint_distance_le_1_min": 8,
        "all_required": True,
        "on_fail": "Do not train v400 or a densified derivative.",
        "on_pass": "Authorize one public-train-only densified integration and offline recursive audit; still no policy update until that audit passes.",
    },
    "evidence_sha256": {name: sha(path) for name, path in paths.items()},
    "guards": {"policy_modified": False, "world_model_rgb_modified_vs_v400": False, "outcomes_used_for_candidate_fit": False, "hidden_or_final_data": False, "real_submission": False},
}
output.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY

cleanup() {
  "$PY" -m ray.scripts.scripts stop --force >/dev/null 2>&1 || true
  TRACK2_SERVICE_REG="$REG" bash "$P/restart_v404_dense_diagnostic_services.sh" stop >/dev/null 2>&1 || true
  bash "$P/restart_v218_services.sh" start >"$REG/restore_v218.log" 2>&1 || true
}
trap cleanup EXIT

TRACK2_SERVICE_REG="$REG" TRACK2_CPUSET=0-5 \
  bash "$P/restart_v404_dense_diagnostic_services.sh" start \
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
  runner.logger.log_path="$RUN" runner.ckpt_path="$V169" actor.seed=1562 \
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
rows = [item for line in trace_path.read_text().splitlines() for item in json.loads(line)["batch"]]
broad = [row for row in rows if row["broad_eligible"]]
phase = np.asarray([row["dense_phase_probability"] for row in broad], dtype=float)
endpoint = np.asarray([row["nearest_endpoint_distance"] for row in broad], dtype=float)
action = np.asarray([row["action_probability"] for row in broad], dtype=float)
q = (0, .1, .25, .5, .75, .9, .95, .99, 1)
quantiles = lambda values: {str(value): float(np.quantile(values, value)) for value in q}
gate = prereg["decision"]
checks = {
    "exact_requests": len(rows) == 800,
    "zero_policy_updates": raw["protocol"]["policy_updates"] == 0,
    "broad_eligible": len(broad) >= gate["broad_eligible_min"],
    "phase_probability_ge_0p5": int((phase >= .5).sum()) >= gate["phase_probability_ge_0p5_min"],
    "endpoint_distance_le_1": int((endpoint <= 1).sum()) >= gate["endpoint_distance_le_1_min"],
}
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
result = {
    "format": "strict-track2-v404-v400-dense-trace32-result-v1",
    "passed": all(checks.values()),
    "requests": len(rows),
    "broad_eligible": len(broad),
    "strict_eligible": sum(row["strict_eligible"] for row in rows),
    "supported_v400_routes": sum(row["supported_v400_route"] for row in rows),
    "phase_probability_ge_0p5": int((phase >= .5).sum()),
    "endpoint_distance_le_1": int((endpoint <= 1).sum()),
    "action_probability_quantiles": quantiles(action),
    "phase_probability_quantiles": quantiles(phase),
    "endpoint_distance_quantiles": quantiles(endpoint),
    "checks": checks,
    "rollout_metrics_context_only_not_selection": raw["metrics"],
    "evidence_sha256": {"preregistration": sha(prereg_path), "raw": sha(raw_path), "trace": sha(trace_path)},
    "guards": {"policy_updates": 0, "checkpoint_writes": 0, "v400_rgb_output_equivalent": True, "hidden_or_final_data": False, "real_submission": False},
}
output_path.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY
