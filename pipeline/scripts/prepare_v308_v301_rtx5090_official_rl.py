#!/usr/bin/env python3
"""Preregister the fresh RTX 5090 continuation of official-policy v301 RL."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


B = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
O = B / "artifacts/strict_track2_official_20260810"
J = B / "artifacts/strict_track2_joint_augmentation_20260810"
N = "v308_v301_rtx5090_fresh_official_h200_r4_step10_lr2e5_beta001_seed1471_20260821"
REG = O / "run_registry" / N
RUN = O / "runs" / N
GATE = J / "v303_v301_batched_gates_seed1482_20260821"
PREVIOUS = "v304_v301_fresh_official_h200_r4_step10_lr2e5_beta001_seed1471_20260821"
DIAGNOSTICS = O / "diagnostics"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if REG.exists() or RUN.exists():
        raise FileExistsError("refusing overwrite")

    capture = GATE / "audit/capture_relative_gate.json"
    long_gate = GATE / "audit/long_gate_report.json"
    if not json.loads(capture.read_text())["passed"]:
        raise RuntimeError("v303 capture gate is not passed")
    if not json.loads(long_gate.read_text())["passed"]:
        raise RuntimeError("v303 long gate is not passed")

    raw_numeric = DIAGNOSTICS / "v308_v301_rtx5090_batch_equivalence_20260821.json"
    fixed_numeric = DIAGNOSTICS / "v308_v301_rtx5090_batch_equivalence_envtf32off_20260821.json"
    service_acceptance = DIAGNOSTICS / "v308_rtx5090_service_acceptance_20260821.json"
    raw = json.loads(raw_numeric.read_text())
    fixed = json.loads(fixed_numeric.read_text())
    service = json.loads(service_acceptance.read_text())
    if not service["passed"]:
        raise RuntimeError("RTX 5090 service acceptance failed")
    numeric_thresholds = {
        "max_absolute_pixel_change": 2,
        "mean_absolute_pixel_change": 0.05,
        "minimum_speedup": 1.5,
    }
    numeric_checks = {
        "unmitigated_blackwell_exceeds_frozen_gate": (
            raw["max_absolute_pixel_change"] > numeric_thresholds["max_absolute_pixel_change"]
            or raw["mean_absolute_pixel_change"] > numeric_thresholds["mean_absolute_pixel_change"]
        ),
        "tf32_disabled_max_within_gate": fixed["max_absolute_pixel_change"]
        <= numeric_thresholds["max_absolute_pixel_change"],
        "tf32_disabled_mean_within_gate": fixed["mean_absolute_pixel_change"]
        <= numeric_thresholds["mean_absolute_pixel_change"],
        "tf32_disabled_speedup_within_gate": fixed["speedup"]
        >= numeric_thresholds["minimum_speedup"],
    }
    if not all(numeric_checks.values()):
        raise RuntimeError(f"RTX 5090 numeric gate failed: {numeric_checks}")

    previous_run = O / "runs" / PREVIOUS
    previous_registry = O / "run_registry" / PREVIOUS
    previous_checkpoints = sorted(previous_run.glob("**/checkpoints/global_step_*"))
    if previous_checkpoints:
        raise RuntimeError("v304 unexpectedly contains a recoverable checkpoint")

    official = B / "artifacts/official_resources/pi05_adjust_bottle/model.safetensors"
    reference = O / "immutable_reference_policy_state/pi05_official_rank0.pt"
    manifest = GATE / "release_registration.json"
    runner = B / "pipeline/scripts/run_strict_track2_conservative_kl.sh"
    actor = B / "third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark/rlinf/workers/actor/fsdp_actor_worker.py"
    embodied = B / "third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark/rlinf/runners/embodied_runner.py"
    openpi = B / "third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark/rlinf/models/embodiment/openpi/openpi_action_model.py"
    runtime = B / "pipeline/wam_pipeline/v301_batched_terminal_frame_mirror_runtime.py"
    backends = B / "pipeline/wam_pipeline/backends.py"
    generic_launcher = B / "pipeline/scripts/launch_v300_v295_official_rl.sh"
    wrapper = B / "pipeline/scripts/launch_v308_v301_rtx5090_official_rl.sh"
    service_wrapper = B / "pipeline/scripts/restart_v308_rtx5090_services.sh"
    governor = B / "pipeline/scripts/govern_v308_cpu_affinity.sh"
    expected_openpi = "b3160f04f1e65f99c9cb75a39c9448586fe8143f5af7462c317580cad4e391ed"
    if sha(openpi) != expected_openpi:
        raise RuntimeError("official fixed OpenPI source is not restored")

    REG.mkdir(parents=True)
    payload = {
        "format": "strict-track2-v308-v301-rtx5090-fresh-official-rl-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "run_path": str(RUN),
        "migration": {
            "predecessor": PREVIOUS,
            "predecessor_preregistration_sha256": sha(previous_registry / "preregistration.json"),
            "predecessor_checkpoint_count": 0,
            "restart_reason": "server migration before the first recovery checkpoint",
            "training_hyperparameters_changed": False,
            "predecessor_training_metrics_used_for_selection": False,
        },
        "hardware": {
            "gpu": "NVIDIA GeForce RTX 5090 32607 MiB",
            "cgroup_cpu_quota_cores": 25,
            "cgroup_memory_max_bytes": 96636764160,
            "project_cpu_affinity": "0-21",
            "reserved_control_plane_cores": 3,
            "tf32_disabled": True,
            "tf32_environment": {
                "NVIDIA_TF32_OVERRIDE": "0",
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            },
            "numeric_thresholds_inherited_from_v303": numeric_thresholds,
            "numeric_checks": numeric_checks,
            "unmitigated_numeric_report_sha256": sha(raw_numeric),
            "tf32_disabled_numeric_report_sha256": sha(fixed_numeric),
            "service_acceptance_sha256": sha(service_acceptance),
            "service_restart_pixel_hash": service["restart_pixel_hash"],
        },
        "world_model": {
            "model_version": "track2-v301-batched-terminal-frame-mirror-v295",
            "manifest": str(manifest),
            "manifest_sha256": sha(manifest),
            "capture_gate_sha256": sha(capture),
            "long_gate_sha256": sha(long_gate),
            "runtime_sha256": sha(runtime),
            "backends_sha256": sha(backends),
        },
        "frozen_training": {
            "initial_policy": "official unmodified Pi0.5 adjust_bottle",
            "initial_policy_path": str(official),
            "initial_policy_sha256": sha(official),
            "canonical_kl_reference": str(reference),
            "canonical_kl_reference_sha256": sha(reference),
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
            "sft_co_training": False,
            "policy_mirroring": False,
            "actor_enable_offload": True,
        },
        "recovery_protocol": {
            "intermediate_save_interval": 3,
            "keep_last_checkpoints": 2,
            "resume_requires_model_optimizer_scheduler_and_global_step": True,
            "resume_preserves_canonical_reference": True,
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
            "embodied_runner_sha256": sha(embodied),
            "openpi_action_model_sha256": sha(openpi),
            "generic_launcher_sha256": sha(generic_launcher),
            "wrapper_sha256": sha(wrapper),
            "service_wrapper_sha256": sha(service_wrapper),
            "cpu_governor_sha256": sha(governor),
        },
        "guards": {
            "hidden_or_final_data_used_for_training_or_selection": False,
            "policy_action_injection": False,
            "participant_modifies_only_world_model": True,
            "official_rule_compliant": True,
            "real_submission": False,
        },
    }
    (REG / "preregistration.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"registry": str(REG), "run": str(RUN)}, indent=2))


if __name__ == "__main__":
    main()
