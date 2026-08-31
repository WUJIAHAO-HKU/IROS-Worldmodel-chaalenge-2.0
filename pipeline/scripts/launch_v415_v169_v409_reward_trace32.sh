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
NAME='v415_v169_v409_reward_trace32_seed1570_20260823'
REG="$O/run_registry/$NAME"
RUN="$O/runs/$NAME"
TRACE="$RUN/audit/reward_trace"
V169="$O/runs/v169_parent_step5_seed1243_20260813/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
GO1='/root/miniconda3/envs/go1/bin/python'

test ! -e "$REG"
test ! -e "$RUN"
mkdir -p "$REG" "$TRACE"

"$GO1" - "$ROOT" "$REG/preregistration.json" <<'PY'
import datetime, hashlib, json, pathlib, sys
root, output = map(pathlib.Path, sys.argv[1:])
j = root / "artifacts/strict_track2_joint_augmentation_20260810"
o = root / "artifacts/strict_track2_official_20260810"
paths = {
    "v409_runtime": root / "pipeline/wam_pipeline/v409_half_contracted_progressive_runtime.py",
    "trace_runtime": root / "pipeline/wam_pipeline/v414_v409_reward_trace_runtime.py",
    "analyzer": root / "pipeline/scripts/analyze_v415_v409_reward_trace.py",
    "service": root / "pipeline/scripts/restart_v414_v409_reward_trace_services.sh",
    "launcher": root / "pipeline/scripts/launch_v415_v169_v409_reward_trace32.sh",
    "v409_recursive_gate": j / "v409_half_contracted_progressive_seed1567_20260823/audit/recursive_reward_causal.json",
    "v410_route_trace": o / "run_registry/v410_v169_v409_trace32_seed1568_20260823/result.json",
    "v412_public_rejection": o / "run_registry/v412_v411_public_batch00_20260823/result.json",
}
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
if not json.loads(paths["v409_recursive_gate"].read_text()).get("passed"):
    raise RuntimeError("v409 recursive gate failed")
if not json.loads(paths["v410_route_trace"].read_text()).get("passed"):
    raise RuntimeError("v410 route trace failed")
if json.loads(paths["v412_public_rejection"].read_text()).get("accepted"):
    raise RuntimeError("v411 was not rejected")
payload = {
    "format": "strict-track2-v415-v409-reward-trace32-preregistration-v1",
    "registered_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "purpose": "Test whether v409's rankable route strength actually creates rankable official training reward on the frozen v169 distribution.",
    "protocol": {"trajectories": 32, "requests": 800, "group_size": 4, "policy_updates": 0, "checkpoint_writes": 0, "actor_seed": 1570},
    "decision": {
        "progressive_routes_min": 48,
        "positive_reward_delta_fraction_min": 0.55,
        "alpha_reward_delta_correlation_min": 0.20,
        "groups_with_delta_spread_ge_0p05_min": 12,
        "all_required": True,
        "on_fail": "Reject v409 as an effective RL reward environment; do not increase policy steps or learning rate.",
        "on_pass": "Reward signal is real; diagnose propagation/budget before any new policy update.",
    },
    "evidence_sha256": {name: sha(path) for name, path in paths.items()},
    "guards": {"world_model_rgb_exact_v409": True, "policy_modified": False, "simulator_outcomes_used": False, "hidden_or_final_data": False, "real_submission": False},
}
output.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY

cleanup() {
  "$PY" -m ray.scripts.scripts stop --force >/dev/null 2>&1 || true
  TRACK2_SERVICE_REG="$REG" TRACK2_TRACE_DIR="$TRACE" \
    bash "$P/restart_v414_v409_reward_trace_services.sh" stop >/dev/null 2>&1 || true
  bash "$P/restart_v218_services.sh" start >"$REG/restore_v218.log" 2>&1 || true
}
trap cleanup EXIT

TRACK2_SERVICE_REG="$REG" TRACK2_TRACE_DIR="$TRACE" TRACK2_CPUSET=0-5 \
  bash "$P/restart_v414_v409_reward_trace_services.sh" start \
  >"$REG/restart_trace_before_rollout.log" 2>&1
"$PY" -m ray.scripts.scripts stop --force >"$RUN/ray_stop_before.log" 2>&1 || true
taskset -c 6-21 "$PY" -m ray.scripts.scripts start --head \
  --object-store-memory=4294967296 --disable-usage-stats >"$RUN/ray_start.log" 2>&1

cd "$RLINF"
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 RAYON_NUM_THREADS=2
export TRACK2_ROLLOUT_ONLY_OUTPUT="$RUN/audit/sequestered_raw.json"
REPO_PATH="$RLINF" EMBODIED_PATH="$RLINF/examples/embodiment" \
OPENPI_CKPT_PATH="$OPENPI_CKPT" WAN_ROBOTWIN_ADJUST_BOTTLE_CKPT="$ADAPTER" \
ROBOTWIN_REWARD_MODEL_PATH="$REWARD" T5_MODEL_PATH="$T5" \
PYTHONPATH="$DIFFSYNTH:$OPENPI/packages/openpi-client/src:$OPENPI/src:$RLINF" \
PYTHONHASHSEED=0 RAY_DEDUP_LOGS=0 taskset -c 6-21 "$PY" \
  "$P/trainmode_rollout_only_embodied_agent.py" \
  --config-path "$RLINF/examples/embodiment/config" \
  --config-name wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05 \
  runner.logger.log_path="$RUN" runner.ckpt_path="$V169" actor.seed=1570 \
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

"$PY" -m ray.scripts.scripts stop --force >"$RUN/ray_stop_before_reward.log" 2>&1 || true
TRACK2_SERVICE_REG="$REG" TRACK2_TRACE_DIR="$TRACE" \
  bash "$P/restart_v414_v409_reward_trace_services.sh" stop >/dev/null 2>&1 || true
cd "$ROOT"
set +e
PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts:$RLINF" \
  OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 taskset -c 0-11 "$GO1" \
  "$P/analyze_v415_v409_reward_trace.py" \
  --trace-dir "$TRACE" --preregistration "$REG/preregistration.json" \
  --reward-checkpoint "$REWARD" --t5-model "$T5" \
  --output "$RUN/audit/reward_signal_analysis.json" --device cuda --batch-size 32 \
  >"$RUN/audit/reward_signal_analysis.log" 2>&1
status=$?
set -e
test -s "$RUN/audit/reward_signal_analysis.json"
cp "$RUN/audit/reward_signal_analysis.json" "$REG/reward_signal_analysis.json"
cp "$RUN/audit/reward_signal_analysis.log" "$REG/reward_signal_analysis.log"
cat "$REG/reward_signal_analysis.log"
exit "$status"
