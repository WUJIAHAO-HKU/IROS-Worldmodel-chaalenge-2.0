#!/usr/bin/env python3
"""Preregister v311 before fitting the public-train-only action gate."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


B = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
O = B / "artifacts/strict_track2_official_20260810"
J = B / "artifacts/strict_track2_joint_augmentation_20260810"
N = "v311_public_action_causal_gate_seed1484_20260822"
RUN = J / N
REG = O / "run_registry" / N


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if RUN.exists() or REG.exists():
        raise FileExistsError("refusing overwrite")
    trainer = B / "pipeline/scripts/train_v311_public_action_causal_gate.py"
    split = J / "v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json"
    source_split = B / "artifacts/splits/adjust_bottle_50episodes_full.json"
    for path in (trainer, split, source_split):
        if not path.is_file():
            raise FileNotFoundError(path)
    payload = {
        "format": "strict-track2-v311-public-action-causal-gate-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "objective": "learn whether a requested right-arm chunk resembles a physically successful public expert transport, without using reward or evaluation outcomes",
        "fixed_training": {
            "positive_source": "right-arm closed-gripper windows from frozen public train40",
            "positive_episode_count": 15,
            "negative_transforms": ["open_gripper", "static_transport", "reverse_transport"],
            "max_positive_windows_per_episode": 128,
            "classifier": "balanced logistic regression",
            "c_candidates": [0.01, 0.1, 1.0, 10.0],
            "threshold_candidates": [round(0.1 + 0.05 * index, 2) for index in range(17)],
            "selection": "maximize minimum grouped-OOF TPR/TNR, then balanced accuracy",
            "grouping": "leave one source episode out",
            "seed": 1484,
        },
        "fixed_acceptance": {
            "grouped_oof_tpr_min": 0.80,
            "grouped_oof_tnr_min": 0.80,
            "all_checks_required": True,
            "waivers_allowed": False,
        },
        "evidence": {str(path): sha256(path) for path in (trainer, split, source_split)},
        "guards": {
            "participant_component": "world-model RGB predictor only",
            "policy_modified": False,
            "official_reward_modified": False,
            "official_rl_algorithm_or_budget_modified": False,
            "validation_or_local_test_used_for_training": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    RUN.mkdir(parents=True)
    (RUN / "audit").mkdir()
    REG.mkdir(parents=True)
    text = json.dumps(payload, indent=2) + "\n"
    (RUN / "release_registration.json").write_text(text)
    (REG / "preregistration.json").write_text(text)
    print(json.dumps({"run": str(RUN), "registry": str(REG)}, indent=2))


if __name__ == "__main__":
    main()
