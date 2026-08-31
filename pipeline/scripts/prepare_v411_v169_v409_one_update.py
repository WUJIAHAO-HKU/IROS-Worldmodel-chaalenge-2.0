#!/usr/bin/env python3
"""Preregister one official conservative update from v169 under frozen v409."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


B = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
O = B / "artifacts/strict_track2_official_20260810"
J = B / "artifacts/strict_track2_joint_augmentation_20260810"
N = "v411_v169step5_v409_oneupdate_h200_r4_lr1e5_seed1569_20260823"
REG = O / "run_registry" / N
RUN = Path("/dev/shm") / N
V169 = O / "runs/v169_parent_step5_seed1243_20260813/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt"
V169_SHA = "41f0bee86472d22cbb6ab6a7d0060f08f0ba0aad93d260582a5f7f078bca776b"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if REG.exists() or RUN.exists():
        raise FileExistsError("refusing overwrite")
    paths = {
        "v409_manifest": J / "v409_half_contracted_progressive_seed1567_20260823/release_registration.json",
        "v409_recursive_gate": J / "v409_half_contracted_progressive_seed1567_20260823/audit/recursive_reward_causal.json",
        "v409_trace32": O / "run_registry/v410_v169_v409_trace32_seed1568_20260823/result.json",
        "canonical_reference": O / "immutable_reference_policy_state/pi05_official_rank0.pt",
        "runner": B / "pipeline/scripts/run_strict_track2_conservative_kl.sh",
        "service_wrapper": B / "pipeline/scripts/restart_v409_services.sh",
        "launch_wrapper": B / "pipeline/scripts/launch_v411_v169_v409_one_update.sh",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    for key in ("v409_recursive_gate", "v409_trace32"):
        if json.loads(paths[key].read_text()).get("passed") is not True:
            raise RuntimeError(f"{key} did not pass")
    if V169.stat().st_size != 8529316588 or not zipfile.is_zipfile(V169) or sha(V169) != V169_SHA:
        raise RuntimeError("v169 integrity failed")

    payload = {
        "format": "strict-track2-v411-v169-v409-one-update-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "run_path": str(RUN),
        "persistent_registry": str(REG),
        "purpose": "single low-cost official update from reproducible v169 51/128 champion under frozen, causally accepted v409",
        "world_model": {
            "model_version": "track2-v409-v407-half-contracted-progressive",
            "manifest_sha256": sha(paths["v409_manifest"]),
            "recursive_gate_sha256": sha(paths["v409_recursive_gate"]),
            "zero_update_trace32_sha256": sha(paths["v409_trace32"]),
        },
        "frozen_training": {
            "initial_policy": "v169 step5 champion historical 51/128",
            "initial_policy_path": str(V169),
            "initial_policy_sha256": V169_SHA,
            "optimizer": "fresh official AdamW",
            "canonical_kl_reference": str(paths["canonical_reference"]),
            "canonical_kl_reference_sha256": sha(paths["canonical_reference"]),
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
            "actor_seed": 1569,
            "env_seed": 0,
            "sft_co_training": False,
            "policy_mirroring": False,
            "actor_enable_offload": True,
        },
        "resource_guards": {
            "volatile_run_storage": "/dev/shm; no historical checkpoint deletion",
            "minimum_free_shm_bytes": 30000000000,
            "service_cpu_affinity": "0-5",
            "rl_cpu_affinity": "6-21",
            "reserved_cores": "22-24",
            "terminal_model_only_checkpoint": True,
            "keep_last_checkpoints": 1,
            "automatic_retry": False,
        },
        "acceptance_gates": {
            "action_dim_normalized_approx_kl_abs_max": 0.01,
            "action_dim_normalized_clip_fraction_max": 0.05,
            "gradient_norm_max": 5.0,
            "finite_nonnegative_kl": True,
            "terminal_checkpoint_zip_integrity": True,
        },
        "selection_protocol": {
            "exactly_one_update": True,
            "no_training_continuation_without_public_batch00": True,
            "one_update_public_batch00_requires_new_preregistration": True,
            "batch01_forbidden_until_batch00_passes": True,
            "reserved_final128_forbidden_until_unique_candidate_freeze": True,
            "trace32_rollout_success_is_context_only_not_selection": True,
            "real_submission": False,
        },
        "evidence": {key: {"path": str(path), "sha256": sha(path)} for key, path in paths.items()},
        "guards": {
            "hidden_or_final_data_used": False,
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
