#!/usr/bin/env python3
"""Preregister the compliant v237 GRPO test against gated v236."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


BASE = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
OFF = BASE / "artifacts/strict_track2_official_20260810"
JOINT = BASE / "artifacts/strict_track2_joint_augmentation_20260810"
NAME = "v237_v236_conservativekl_h200_r2_step8_lr5e6_beta005_seed1436_20260818"
REG = OFF / "run_registry" / NAME
RUN = OFF / "runs" / NAME
RELEASE = JOINT / "v236_v235c_causal_right_close_service_seed1435_20260818"
V236_AUDIT = RELEASE / "audit/v236_causal_service_report.json"
V209 = JOINT / "v209_v202_v208_public_arm_routed_release"
V214 = JOINT / "v214_public_right_knn_action_visual_diagnostic_seed1413"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if (REG / "preregistration.json").exists() or RUN.exists():
        raise SystemExit("refusing to overwrite v237 preregistration or run")
    audit = json.loads(V236_AUDIT.read_text())
    if audit.get("passed") is not True:
        raise ValueError("v236 causal online service audit did not pass")
    if not (RELEASE / "V236_PUBLIC_GATE_PASSED").exists():
        raise ValueError("v236 public gate marker is absent")
    REG.mkdir(parents=True, exist_ok=True)
    manifest = RELEASE / "release_registration.json"
    release_doc = json.loads(manifest.read_text())
    left = V209 / "left_expert/model.pt"
    right = V209 / "right_expert/model.pt"
    library = V214 / "library/public_right_knn.npz"
    runtime = BASE / "pipeline/wam_pipeline/v236_causal_public_knn_runtime.py"
    runner = BASE / "pipeline/scripts/run_strict_track2_conservative_kl.sh"
    payload = {
        "format": "strict-track2-v237-v236-conservative-kl-preregistration-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "test whether the preregistered causal right-close world-model correction "
            "improves official fixed-budget GRPO while preserving the audited v210 configuration"
        ),
        "run_path": str(RUN),
        "world_model": {
            "model_version": release_doc["model_version"],
            "right_blend_alpha": release_doc["alpha"],
            "causal_gate": release_doc["causal_gate"],
            "release": str(RELEASE),
            "manifest": str(manifest),
            "manifest_sha256": sha256(manifest),
            "left_model_sha256": sha256(left),
            "right_model_sha256": sha256(right),
            "public_online_service_audit": str(V236_AUDIT),
            "public_online_service_audit_sha256": sha256(V236_AUDIT),
            "public_library_sha256": sha256(library),
            "runtime_sha256": sha256(runtime),
        },
        "change_from_v219": {
            "only_intended_algorithmic_change": "unconditional right-route RGB retrieval -> causal future right-gripper-close retrieval",
            "actor_seed": "new deterministic seed 1436 for a new run",
            "unchanged": [
                "fresh official Pi0.5 initialization",
                "8 updates and 2x200 rollout",
                "actor learning rate 5e-6",
                "KL beta 0.05 and action-normalized low_var_kl",
                "group size 4 and total environments 8",
            ],
        },
        "frozen_training": {
            "initial_policy": "official unmodified Pi0.5 adjust_bottle checkpoint",
            "max_steps": 8,
            "episode_steps": 200,
            "rollout_steps": 200,
            "rollout_epoch": 2,
            "total_envs": 8,
            "group_size": 4,
            "actor_global_batch_size": 400,
            "actor_lr": 5e-6,
            "kl_beta": 0.05,
            "kl_penalty": "low_var_kl",
            "actor_seed": 1436,
            "env_seed": 0,
            "reference_state_storage": "disk_mmap_during_kl_only",
            "terminal_checkpoint_contents": "full_model_weights_only",
        },
        "acceptance_gates": {
            "action_dim_normalized_approx_kl_abs_max": 0.01,
            "action_dim_normalized_clip_fraction_max": 0.05,
            "gradient_norm_max": 5.0,
            "all_update_kl_loss_finite_nonnegative": True,
            "checkpoint_zip_integrity_required": True,
        },
        "public_policy_screen": {
            "stage_a": "19/32, +2 over baseline, both arms >=50%, grasps not below baseline",
            "stage_b": "75/112, +3 over baseline, both arms >=60%",
            "selection_data": "frozen public unseen train seeds only",
        },
        "runner": {"path": str(runner), "sha256": sha256(runner)},
        "guards": {
            "real_submission": False,
            "final_128_used_for_training_or_selection": False,
            "fixed_official_initial_policy": True,
            "policy_action_injection": False,
        },
        "prohibited_inputs": [
            "hidden development outcomes",
            "official final outcomes",
            "real contest submission feedback",
        ],
    }
    path = REG / "preregistration.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"registry": str(REG), "run": str(RUN)}, indent=2))


if __name__ == "__main__":
    main()
