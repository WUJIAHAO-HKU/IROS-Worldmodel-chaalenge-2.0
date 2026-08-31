#!/usr/bin/env python3
"""Preregister one official RL update from the frozen v169 champion under v326."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


B = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
O = B / "artifacts/strict_track2_official_20260810"
J = B / "artifacts/strict_track2_joint_augmentation_20260810"
N = "v327_v169step5_v326_authorized_oneupdate_h200_r4_lr1e5_seed1497_20260822"
REG = O / "run_registry" / N
RUN = O / "runs" / N
V169 = O / "runs/v169_parent_step5_seed1243_20260813/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt"
V169_SHA = "41f0bee86472d22cbb6ab6a7d0060f08f0ba0aad93d260582a5f7f078bca776b"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if REG.exists() or RUN.exists():
        raise FileExistsError("refusing overwrite")
    paths = {
        "v326_manifest": J / "v326_v317_blended_phase_terminal_gate_seed1496_20260822/release_registration.json",
        "v326_authorization": J / "v326_v317_blended_phase_terminal_gate_seed1496_20260822/audit/expensive_rl_authorization.json",
        "v326_service_acceptance": J / "v326_v317_blended_phase_terminal_gate_seed1496_20260822/audit/service_acceptance.json",
        "v169_champion": V169,
        "canonical_reference": O / "immutable_reference_policy_state/pi05_official_rank0.pt",
        "runner": B / "pipeline/scripts/run_strict_track2_conservative_kl.sh",
        "service_wrapper": B / "pipeline/scripts/restart_v326_services.sh",
        "launch_wrapper": B / "pipeline/scripts/launch_v327_v169_v326_one_update.sh",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    authorization = json.loads(paths["v326_authorization"].read_text())
    if authorization.get("passed") is not True or authorization.get("launch_permission") is not True:
        raise RuntimeError("v326 RL authorization failed")
    if json.loads(paths["v326_service_acceptance"].read_text()).get("passed") is not True:
        raise RuntimeError("v326 service acceptance failed")
    if V169.stat().st_size != 8_529_316_588 or not zipfile.is_zipfile(V169):
        raise RuntimeError("v169 champion checkpoint integrity failed")
    if sha256(V169) != V169_SHA:
        raise RuntimeError("v169 champion checkpoint hash mismatch")
    payload = {
        "format": "strict-track2-v327-v169-v326-one-update-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "run_path": str(RUN),
        "purpose": "low-cost go/no-go update from the reproducible 51/128 v169 champion; do not launch a long run until training-only evidence and a separately preregistered public gate justify it",
        "world_model": {
            "model_version": "track2-v326-blended-phase-terminal-v317",
            "authorization_sha256": sha256(paths["v326_authorization"]),
            "service_acceptance_sha256": sha256(paths["v326_service_acceptance"]),
        },
        "frozen_training": {
            "initial_policy": "frozen v169 step5 champion (historical 51/128)",
            "initial_policy_path": str(V169),
            "initial_policy_sha256": V169_SHA,
            "optimizer": "new official AdamW state because v169 step5 has no optimizer checkpoint",
            "canonical_kl_reference": str(paths["canonical_reference"]),
            "canonical_kl_reference_sha256": sha256(paths["canonical_reference"]),
            "max_steps": 1,
            "trajectories_per_update": 128,
            "total_accepted_trajectories": 128,
            "episode_steps": 200,
            "rollout_steps": 200,
            "rollout_epoch": 4,
            "total_envs": 32,
            "group_size": 4,
            "actor_global_batch_size": 3200,
            "actor_lr": 1e-5,
            "kl_beta": 0.01,
            "kl_penalty": "low_var_kl",
            "actor_seed": 1497,
            "env_seed": 0,
            "sft_co_training": False,
            "policy_mirroring": False,
            "actor_enable_offload": True,
        },
        "resource_guards": {
            "minimum_free_data_disk_bytes_at_launch": 16_000_000_000,
            "service_cpu_affinity": "0-5",
            "rl_cpu_affinity": "6-21",
            "reserved_control_plane_cores": "22-24",
            "terminal_model_only_checkpoint": True,
            "keep_last_checkpoints": 1,
            "automatic_multi_attempt_retry": False,
        },
        "acceptance_gates": {
            "action_dim_normalized_approx_kl_abs_max": 0.01,
            "action_dim_normalized_clip_fraction_max": 0.05,
            "gradient_norm_max": 5.0,
            "all_update_kl_loss_finite_nonnegative": True,
            "terminal_checkpoint_zip_integrity_required": True,
        },
        "selection_protocol": {
            "training_metrics_only_before_candidate_freeze": True,
            "initial_rollout_success_rate_go_signal": 0.25,
            "comparison_baseline_v318_step0_success_rate": 0.078125,
            "one_update_checkpoint_public_batch00_requires_new_preregistration": True,
            "batch01_forbidden_until_batch00_passes": True,
            "reserved_final128_forbidden_until_unique_candidate_freeze": True,
            "real_submission": False,
        },
        "evidence": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items() if name != "v169_champion"},
        "guards": {
            "hidden_or_final_data_used_for_training_or_selection": False,
            "policy_action_injection": False,
            "participant_modifies_only_world_model": True,
            "official_rl_algorithm": True,
            "real_submission": False,
        },
    }
    REG.mkdir(parents=True)
    (REG / "preregistration.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"registry": str(REG), "run": str(RUN)}, indent=2))


if __name__ == "__main__":
    main()
