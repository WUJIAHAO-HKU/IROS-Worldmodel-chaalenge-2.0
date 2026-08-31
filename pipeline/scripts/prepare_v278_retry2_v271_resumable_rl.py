#!/usr/bin/env python3
"""Preregister the resumable v278 conservative RL run before any evaluation."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

B = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
O = B / "artifacts/strict_track2_official_20260810"
J = B / "artifacts/strict_track2_joint_augmentation_20260810"
N = "v278_v271_fresh_fullbudget_h200_r4_step10_lr2e5_beta001_seed1471_retry2_20260820"
REG = O / "run_registry" / N
RUN = O / "runs" / N
GATE = J / "v273_v271_corrected_alpha_long_gate_seed1470_20260819"
MANIFEST = GATE / "release_registration.json"
RETRY1 = O / "runs/v274_v271_fresh_fullbudget_h200_r8_step5_lr2e5_beta001_seed1471_retry1_20260819"
RETRY1_LOG = RETRY1 / "launcher.log"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if REG.exists() or RUN.exists():
        raise FileExistsError("refusing to overwrite an existing retry2 run")

    capture = GATE / "audit/corrected_endpoint_alpha_report.json"
    long_gate = GATE / "audit/long_gate_report.json"
    assert json.loads(capture.read_text())["passed"] is True
    assert json.loads(long_gate.read_text())["passed"] is True
    retry1_text = RETRY1_LOG.read_text(errors="replace")
    assert "keepalive watchdog timeout" in retry1_text
    assert not list(RETRY1.glob("**/checkpoints/**/*.pt"))

    official = B / "artifacts/official_resources/pi05_adjust_bottle/model.safetensors"
    runner = B / "pipeline/scripts/run_strict_track2_conservative_kl.sh"
    actor = (
        B
        / "third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
        / "rlinf/workers/actor/fsdp_actor_worker.py"
    )
    embodied_runner = (
        B
        / "third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
        / "rlinf/runners/embodied_runner.py"
    )
    runtime = B / "pipeline/wam_pipeline/v271_endpoint_calibrated_terminal_runtime.py"
    backends = B / "pipeline/wam_pipeline/backends.py"
    canonical_reference = O / "immutable_reference_policy_state/pi05_official_rank0.pt"
    assert canonical_reference.is_file()

    REG.mkdir(parents=True)
    payload = {
        "format": "strict-track2-v278-v271-resumable-conservative-rl-preregistration-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_path": str(RUN),
        "reason": {
            "retry1_path": str(RETRY1),
            "retry1_log_sha256": sha(RETRY1_LOG),
            "failure": "container-wide approximately 1390-second scheduling pause caused Ray ChannelWorker keepalive timeout during step 2 rollout 7/8",
            "excluded_causes": ["CUDA OOM", "kernel OOM kill", "disk-full write failure", "server reboot"],
        },
        "world_model": {
            "model_version": "track2-v271-endpoint-calibrated-terminal",
            "manifest": str(MANIFEST),
            "manifest_sha256": sha(MANIFEST),
            "corrected_capture_gate_sha256": sha(capture),
            "long_horizon_gate_sha256": sha(long_gate),
            "runtime_sha256": sha(runtime),
            "backends_sha256": sha(backends),
        },
        "frozen_training": {
            "initial_policy": "official unmodified Pi0.5 adjust_bottle",
            "initial_policy_path": str(official),
            "initial_policy_sha256": sha(official),
            "max_steps": 10,
            "trajectories_per_update": 128,
            "total_accepted_trajectories": 1280,
            "episode_steps": 200,
            "rollout_steps": 200,
            "rollout_epoch": 4,
            "total_envs": 32,
            "group_size": 4,
            "actor_global_batch_size": 3200,
            "actor_lr": 2e-5,
            "kl_beta": 0.01,
            "kl_penalty": "low_var_kl",
            "actor_seed": 1471,
            "env_seed": 0,
            "actor_enable_offload": True,
            "reference_state_storage": "immutable official disk state",
        },
        "recovery_protocol": {
            "max_process_attempts": 4,
            "intermediate_save_interval": 1,
            "keep_last_checkpoints": 2,
            "checkpoint_contents": ["full model weights", "CPU optimizer state", "LR scheduler state", "global step"],
            "resume_preserves_official_kl_reference": True,
            "ray_grpc_keepalive_timeout_ms": 1800000,
            "ray_health_check_timeout_ms": 1800000,
            "rollout_epoch_change": "8 to 4; max updates 5 to 10; total trajectories and total action-chunk samples unchanged",
        },
        "acceptance_gates": {
            "action_dim_normalized_approx_kl_abs_max": 0.01,
            "action_dim_normalized_clip_fraction_max": 0.05,
            "gradient_norm_max": 5.0,
            "all_update_kl_loss_finite_nonnegative": True,
            "terminal_checkpoint_zip_integrity_required": True,
        },
        "selection_protocol": {
            "training_metrics_only_before_candidate_preregistration": True,
            "first_public_gate_after_training": "batch00 only",
            "batch01_forbidden_until_batch00_passes": True,
            "reserved_final128_forbidden_until_unique_candidate_freeze": True,
            "real_submission": False,
        },
        "sources": {
            "runner_sha256": sha(runner),
            "actor_sha256": sha(actor),
            "embodied_runner_sha256": sha(embodied_runner),
            "canonical_reference_sha256": sha(canonical_reference),
        },
        "guards": {
            "hidden_or_final_data_used_for_training_or_selection": False,
            "policy_action_injection": False,
            "official_rule_compliant": True,
        },
    }
    (REG / "preregistration.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"registry": str(REG), "run": str(RUN)}, indent=2))


if __name__ == "__main__":
    main()
