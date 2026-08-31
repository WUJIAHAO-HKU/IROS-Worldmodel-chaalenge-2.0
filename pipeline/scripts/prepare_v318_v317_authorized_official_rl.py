#!/usr/bin/env python3
"""Preregister the only authorized fixed-budget RL run for v317."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


B = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
O = B / "artifacts/strict_track2_official_20260810"
J = B / "artifacts/strict_track2_joint_augmentation_20260810"
N = "v318_v317_authorized_official_h200_r4_step10_lr2e5_beta001_seed1471_20260822"
REG = O / "run_registry" / N
RUN = O / "runs" / N
RELEASE = J / "v317_v315_native_batch_causal_gate_seed1490_20260822"
AUTH_SHA = "a8dec8fdb42b8a2ee030c3127787bff0ca6fabefe49041c3e3a36d436887fa7f"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if REG.exists() or RUN.exists():
        raise FileExistsError("refusing overwrite")
    manifest = RELEASE / "release_registration.json"
    contract = RELEASE / "audit/contract_report.json"
    causal = RELEASE / "audit/causal_gate_report.json"
    authorization = RELEASE / "audit/expensive_rl_authorization.json"
    for path in (manifest, contract, causal, authorization):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha(authorization) != AUTH_SHA:
        raise RuntimeError("authorization hash mismatch")
    for path in (contract, causal, authorization):
        report = json.loads(path.read_text())
        if report.get("passed") is not True:
            raise RuntimeError(f"gate not passed: {path}")
    if json.loads(authorization.read_text()).get("launch_permission") is not True:
        raise RuntimeError("RL launch not authorized")

    official = B / "artifacts/official_resources/pi05_adjust_bottle/model.safetensors"
    reference = O / "immutable_reference_policy_state/pi05_official_rank0.pt"
    runner = B / "pipeline/scripts/run_strict_track2_conservative_kl.sh"
    actor = B / "third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark/rlinf/workers/actor/fsdp_actor_worker.py"
    embodied = B / "third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark/rlinf/runners/embodied_runner.py"
    openpi = B / "third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark/rlinf/models/embodiment/openpi/openpi_action_model.py"
    runtime = B / "pipeline/wam_pipeline/v317_batched_sparse_failure_terminal_runtime.py"
    backends = B / "pipeline/wam_pipeline/backends.py"
    generic_launcher = B / "pipeline/scripts/launch_v300_v295_official_rl.sh"
    wrapper = B / "pipeline/scripts/launch_v318_v317_official_rl.sh"
    service_wrapper = B / "pipeline/scripts/restart_v317_services.sh"
    expected_openpi = "b3160f04f1e65f99c9cb75a39c9448586fe8143f5af7462c317580cad4e391ed"
    if sha(openpi) != expected_openpi:
        raise RuntimeError("official fixed OpenPI source is not restored")

    REG.mkdir(parents=True)
    payload = {
        "format": "strict-track2-v318-v317-authorized-official-rl-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "run_path": str(RUN),
        "predecessor_disposition": {
            "v308_policy_candidate": "rejected after public batch00 total=1/16 right=0/12",
            "v308_checkpoint_reused": False,
            "initial_policy_reset_to_official": True,
            "v308_training_metrics_used_for_v318_selection": False,
        },
        "hardware": {
            "gpu": "NVIDIA GeForce RTX 5090 32607 MiB",
            "cgroup_cpu_quota_cores": 25,
            "cgroup_memory_max_bytes": 96636764160,
            "service_cpu_affinity": "0-5",
            "rl_cpu_affinity": "6-21",
            "reserved_control_plane_cores": "22-24",
            "maximum_project_cpu_share": "22/25 cores by affinity, with per-library thread caps",
            "tf32_disabled": True,
        },
        "world_model": {
            "model_version": "track2-v317-batched-sparse-failure-terminal-v315",
            "manifest": str(manifest),
            "manifest_sha256": sha(manifest),
            "contract_sha256": sha(contract),
            "causal_gate_sha256": sha(causal),
            "authorization_sha256": sha(authorization),
            "authorization_expected_sha256": AUTH_SHA,
            "runtime_sha256": sha(runtime),
            "backends_sha256": sha(backends),
            "native_batch_size": 8,
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
