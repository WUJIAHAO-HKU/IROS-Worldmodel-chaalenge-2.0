#!/usr/bin/env python3
"""Preregister fresh fixed-budget official-policy RL against frozen v295."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

B = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
O = B / "artifacts/strict_track2_official_20260810"
J = B / "artifacts/strict_track2_joint_augmentation_20260810"
N = "v300_v295_fresh_official_h200_r4_step10_lr2e5_beta001_seed1471_20260821"
REG = O / "run_registry" / N
RUN = O / "runs" / N
GATE = J / "v299_v295_corrected_numeric_long_gate_seed1481_20260821"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if REG.exists() or RUN.exists():
        raise FileExistsError("refusing overwrite")
    capture = GATE / "audit/capture_relative_gate.json"
    long_gate = GATE / "audit/long_gate_report.json"
    assert json.loads(capture.read_text())["passed"] is True
    assert json.loads(long_gate.read_text())["passed"] is True
    official = B / "artifacts/official_resources/pi05_adjust_bottle/model.safetensors"
    reference = O / "immutable_reference_policy_state/pi05_official_rank0.pt"
    manifest = GATE / "release_registration.json"
    runner = B / "pipeline/scripts/run_strict_track2_conservative_kl.sh"
    actor = B / "third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark/rlinf/workers/actor/fsdp_actor_worker.py"
    embodied = B / "third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark/rlinf/runners/embodied_runner.py"
    openpi = B / "third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark/rlinf/models/embodiment/openpi/openpi_action_model.py"
    runtime = B / "pipeline/wam_pipeline/v295_terminal_frame_preserving_mirror_runtime.py"
    backends = B / "pipeline/wam_pipeline/backends.py"
    expected_openpi = "b3160f04f1e65f99c9cb75a39c9448586fe8143f5af7462c317580cad4e391ed"
    if sha(openpi) != expected_openpi:
        raise RuntimeError("official fixed OpenPI source is not restored")
    REG.mkdir(parents=True)
    payload = {
        "format": "strict-track2-v300-v295-fresh-official-rl-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "run_path": str(RUN),
        "world_model": {
            "model_version": "track2-v295-terminal-frame-preserving-mirror-v271",
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
            "resource_affinity_only": "world model 0-15; RL/Ray 16-47",
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
        },
        "guards": {
            "hidden_or_final_data_used_for_training_or_selection": False,
            "policy_action_injection": False,
            "participant_modifies_only_world_model": True,
            "official_rule_compliant": True,
        },
    }
    (REG / "preregistration.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"registry": str(REG), "run": str(RUN)}, indent=2))


if __name__ == "__main__":
    main()
