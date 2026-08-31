#!/usr/bin/env python3
"""Write the immutable v432 short-gate preregistration."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
REGISTRY = ROOT / "artifacts/strict_track2_official_20260810/run_registry/v432_public_mirror_prompt_terminal_seed1582_20260823"
PARENT = ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v354_v353_parametric_right_dynamics_extension_seed1523_20260822/model/checkpoints/checkpoint_step_000150"
SPLIT = ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json"
FILES = {
    "trainer": ROOT / "pipeline/scripts/train_v432_public_mirror_prompt_terminal.py",
    "auditor": ROOT / "pipeline/scripts/audit_v432_public_mirror_contract.py",
    "step25_auditor": ROOT / "pipeline/scripts/audit_v432_step25_shortgate.py",
    "launcher": ROOT / "pipeline/scripts/launch_v432_public_mirror_prompt_terminal_shortgate.sh",
    "base_trainer": ROOT / "pipeline/scripts/train_multichunk_reward_aligned_autoregressive_unet.py",
    "split": SPLIT,
    "parent_model": PARENT / "model.pt",
    "official_reward": ROOT / "artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    REGISTRY.mkdir(parents=True, exist_ok=True)
    output = REGISTRY / "preregistration.json"
    if output.exists():
        raise FileExistsError(output)
    split = json.loads(SPLIT.read_text())
    arms = {int(key): str(value) for key, value in split["arm_by_episode"].items()}
    train = [int(value) for value in split["train_episodes"]]
    validation = [int(value) for value in split["validation_episodes"]]
    if (len(train), len(validation), sum(arms[x] == "right" for x in train)) != (40, 10, 15):
        raise RuntimeError("unexpected fixed public split")
    payload = {
        "format": "strict-track2-v432-public-mirror-prompt-terminal-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "diagnosis": {
            "v423_first8_mae_relative_change_percent": 0.6860699280137661,
            "v423_frames25to32_mae_relative_change_percent": -6.024594116541138,
            "v429_learned_mean_reward_delta": -3.032444493230032e-06,
            "v429_learned_positive_fraction": 0.20382165605095542,
            "interpretation": "v423 selected long-horizon gains while regressing the official first8 consumption window; mirror examples also dominated its sample count.",
        },
        "immutable_data": {
            "source": "fixed official public train40 only",
            "train_episodes": train,
            "validation_episodes": validation,
            "real_right_train_episodes": 15,
            "mirrored_left_train_episodes": 25,
            "outcomes_or_rewards_read_for_sampling": False,
            "development_or_final_data": False,
        },
        "training": {
            "initialization": str(PARENT),
            "steps_initial": 25,
            "steps_max_after_gate": 50,
            "batch_size": 1,
            "learning_rate": 1e-7,
            "recursive_chunks": 4,
            "frames_per_chunk": 8,
            "mixture": {"real_right": 0.70, "mirrored_left_to_right": 0.20, "terminal_real_right": 0.10},
            "first8_visual_weight": 2.0,
            "terminal_visual_weight": 2.0,
            "reward_loss_weight": 0.1,
            "reward_delta_weight": 2.0,
            "reward_terminal_weight": 10.0,
            "reward_model_frozen": True,
            "prompt": "v205 episode prompt; mirror rewrites explicit left arm to explicit right arm; hard action/prompt consistency assertion",
            "seed": 1582,
        },
        "short_gate": {
            "selection": "fixed 32 real-right public holdout sequences stratified across early/grasp/post-grasp/terminal",
            "first8_mae_relative_improvement_min": 0.002,
            "recursive32_mae_regression_max": 0.002,
            "temporal_delta_non_regression": True,
            "official_reward_prediction_mae_relative_improvement_min": 0.02,
            "official_reward_endpoint_mae_relative_improvement_min": 0.02,
            "all_required": True,
            "on_fail": "stop; do not continue to step50 or RL",
            "on_pass": "continue the same preregistered run only to step50, then recursive gate and 8-trajectory zero-update trace",
        },
        "post_step50_gate": {
            "zero_update_trajectories": 8,
            "requests": 200,
            "learned_request_mean_delta_positive": True,
            "learned_request_positive_fraction_min": 0.55,
            "trajectory_positive_fraction_min": 0.50,
            "on_fail": "reject before RL",
        },
        "sha256": {key: sha256(path) for key, path in FILES.items()},
        "guards": {
            "official_pi05_modified": False,
            "official_reward_modified": False,
            "official_rl_modified": False,
            "hidden_or_final_data": False,
            "real_competition_submission": False,
            "policy_training_authorized_by_this_file": False,
        },
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
