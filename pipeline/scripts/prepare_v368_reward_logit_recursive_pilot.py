#!/usr/bin/env python3
"""Pre-register a short public-only right-arm recursive reward-logit pilot."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
JOINT = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
RUN = JOINT / "v368_v354_right_reward_logit_recursive_pilot_seed1532_20260822"
PARENT = JOINT / "v354_v353_parametric_right_dynamics_extension_seed1523_20260822/model/best"
SPLIT = JOINT / "v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json"
BASELINE = JOINT / "v356_v355_parametric_recursive_seed1524_20260822/recursive_gate_report.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    RUN.mkdir(parents=True, exist_ok=True)
    output = RUN / "release_registration.json"
    if output.exists():
        raise FileExistsError(output)

    split = json.loads(SPLIT.read_text())
    right_train = [
        episode
        for episode in split["train_episodes"]
        if split["arm_by_episode"][str(episode)] == "right"
    ]
    right_validation = [
        episode
        for episode in split["validation_episodes"]
        if split["arm_by_episode"][str(episode)] == "right"
    ]
    if right_train != [12, 15, 20, 25, 28, 30, 32, 33, 37, 40, 42, 44, 46, 47, 49]:
        raise RuntimeError("unexpected public right-arm training split")
    if right_validation != [6, 7, 18, 22]:
        raise RuntimeError("unexpected public right-arm validation split")

    sources = {
        "trainer": ROOT / "pipeline/scripts/train_multichunk_reward_aligned_autoregressive_unet.py",
        "audit": ROOT / "pipeline/scripts/audit_v368_reward_logit_recursive_pilot.py",
        "parent_model": PARENT / "model.pt",
        "split": SPLIT,
        "baseline_recursive_report": BASELINE,
        "reward_model": ROOT / "artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt",
    }
    payload = {
        "format": "strict-track2-v368-public-right-reward-logit-recursive-pilot-preregistration-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": (
            "v354 teacher-forced terminal semantics survive, but recursive reward collapses; "
            "a bounded 64-frame public-demo logit objective can restore semantic fidelity "
            "without exceeding one-percent recursive RGB/temporal regression"
        ),
        "initialization": "frozen v354 best right expert (step 300)",
        "training": {
            "steps": 20,
            "checkpoint_interval": 10,
            "batch_size": 1,
            "chunks": 8,
            "chunk_frames": 8,
            "recursive_exposure_frames": 64,
            "arm_filter": "right",
            "learning_rate": 5e-8,
            "reward_objective": "logit",
            "reward_loss_weight": 0.002,
            "reward_delta_weight": 0.5,
            "reward_terminal_weight": 2.0,
            "late_start": 48,
            "late_weight": 12.0,
            "terminal_visual_weight": 2.0,
            "max_grad_norm": 1.0,
            "seed": 1532,
        },
        "selection": {
            "decision_split": "public episode-disjoint validation episodes 7 and 18",
            "confirmation_only_split": "public episode-disjoint local-test episodes 6 and 22",
            "candidate_steps": [10, 20],
            "hard_gate": {
                "validation_recursive_rgb_ratio_max": 1.01,
                "validation_recursive_temporal_ratio_max": 1.01,
                "validation_recursive_reward_mae_ratio_max": 0.95,
            },
            "tie_break": "lowest validation recursive reward MAE ratio, then RGB ratio",
            "no_candidate_passing_means_reject": True,
        },
        "authorization_if_passed": "32-trajectory train-mode rollout only; no RL update and no official evaluation",
        "guards": {
            "public_train_only": True,
            "episode_disjoint_public_validation_only": True,
            "frozen_official_reward_model": True,
            "hidden_or_final_evaluation_data": False,
            "official_batch16_outcomes_used": False,
            "real_submission": False,
        },
        "source_sha256": {name: sha256(path) for name, path in sources.items()},
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
