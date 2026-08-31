#!/usr/bin/env python3
"""Prospectively freeze v441 train-only calibration and S0/S1 gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    split = json.loads(args.split.read_text())
    arms = {int(key): str(value) for key, value in split["arm_by_episode"].items()}
    train = [int(value) for value in split["train_episodes"]]
    validation = [int(value) for value in split["validation_episodes"]]
    right = sorted(episode for episode in train if arms[episode] == "right")
    if len(train) != 40 or len(validation) != 10 or len(right) != 15 or set(train) & set(validation):
        raise RuntimeError("wrong public train40/right15 split")
    files = {
        "calibrator": ROOT / "pipeline/scripts/calibrate_v441_postclose_trainonly_rgb_projection.py",
        "runtime": ROOT / "pipeline/wam_pipeline/v441_v169_postclose_aligned_projection_runtime.py",
        "s0_auditor": ROOT / "pipeline/scripts/audit_v441_static_contract.py",
        "packager": ROOT / "pipeline/scripts/package_v441_postclose_aligned_release.py",
        "split": args.split,
        "v169_manifest": J / "v169_instruction_arm_routed_release/v169_arm_routed_manifest.json",
        "v436_manifest": J / "v436_v432_step25_parent_diagnostic_release/v436_diagnostic_manifest.json",
    }
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    payload = {
        "format": "strict-track2-v441-postclose-trainonly-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "classification": "parent world-model diagnostic only; no RL authority",
        "construction": {
            "fallback": "original v169 RGB",
            "teacher_delta": "per-channel clip(v432-step25 - v354, -8, 8)",
            "true_correction": "public train target RGB - original v169 RGB",
            "phases": ["postclose"],
            "samples_per_right_train_episode_phase": 4,
            "selection": "action-only deterministic evenly spaced rows inside each episode/phase",
            "calibration": "leave one right-train episode out; scalar beta per phase/frame/RGB channel",
            "folds": 15,
            "sign_agreement_min": 0.80,
            "heldout_mse_improvement_fraction_min": 0.80,
            "coefficient_abs_max": 1.0,
            "inactive_component": "zero",
            "developer_alpha_or_threshold_sweep": False,
        },
        "data": {
            "train_episodes": train,
            "right_train_episodes": right,
            "validation_episodes_excluded": validation,
            "reward_or_outcome_for_calibration": False,
            "development_or_final_for_calibration": False,
        },
        "s0_gate": {
            "exact_train40_right15": True,
            "exact_samples": 60,
            "active_component_each_phase_min": 1,
            "every_active_sign_agreement_min": 0.80,
            "every_active_heldout_improvement_fraction_min": 0.80,
            "coefficient_abs_max": 1.0,
            "left_and_g0_bitexact": True,
            "candidate_minus_v169_pixel_abs_max": 8,
            "runtime_reward_free": True,
            "all_required": True,
            "on_fail": "reject before S1",
        },
        "s1_gate": {
            "fixed_right_dev_samples": 32,
            "first8_rgb_mae_ratio_to_v169_max": 0.998,
            "recursive32_rgb_mae_ratio_to_v169_max": 1.002,
            "reward_prediction_mae_ratio_to_v169_max": 1.0,
            "endpoint_reward_prediction_mae_ratio_to_v169_max": 1.0,
            "final_reward_mean_ratio_to_v169_min": 0.95,
            "true_over_phase_shuffle_target_mae_max": 0.998,
            "right_intervention_fraction_min": 0.20,
            "right_intervention_fraction_max": 0.80,
            "candidate_minus_v169_pixel_abs_max": 8,
            "left_and_g0_bitexact": True,
            "all_required": True,
            "on_fail": "reject; no trace and no RL",
            "on_pass": "permit only a separately preregistered zero-update short trace",
        },
        "evidence_sha256": {key: sha256(path) for key, path in files.items()},
        "guards": {
            "official_pi05_modified": False, "official_reward_modified": False,
            "official_reward_runtime_used": False, "policy_updates": 0,
            "hidden_or_final_data": False, "real_submission": False,
            "rl_authorized": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

