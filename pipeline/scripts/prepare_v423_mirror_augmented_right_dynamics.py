#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
REGISTRY = ROOT / "artifacts/strict_track2_official_20260810/run_registry/v423_v354s150_mirror_augmented_right_dynamics_seed1575_20260823"
WORK = Path("/dev/shm/v423_v354s150_mirror_augmented_right_dynamics_seed1575_20260823")
PARENT = ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v354_v353_parametric_right_dynamics_extension_seed1523_20260822/model/checkpoints/checkpoint_step_000150"
SPLIT = ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json"
TRAINER = ROOT / "pipeline/scripts/train_v423_mirror_augmented_autoregressive_unet.py"
MIRROR_AUDIT = ROOT / "artifacts/strict_track2_official_20260810/diagnostics/train40_mirror_augmentation_audit_20260819.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    REGISTRY.mkdir(parents=True, exist_ok=True)
    output = REGISTRY / "preregistration.json"
    if output.exists():
        raise FileExistsError(output)
    split = json.loads(SPLIT.read_text())
    audit = json.loads(MIRROR_AUDIT.read_text())
    if not audit.get("accepted") or audit.get("forbidden_episodes_read"):
        raise RuntimeError("the fixed public mirror contract did not pass")
    arm = {int(key): value for key, value in split["arm_by_episode"].items()}
    train_left = [episode for episode in split["train_episodes"] if arm[episode] == "left"]
    train_right = [episode for episode in split["train_episodes"] if arm[episode] == "right"]
    validation_right = [episode for episode in split["validation_episodes"] if arm[episode] == "right"]
    if (len(train_left), len(train_right), len(validation_right)) != (25, 15, 4):
        raise RuntimeError("unexpected fixed public arm counts")
    payload = {
        "format": "strict-track2-v423-mirror-augmented-right-dynamics-preregistration-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": (
            "v354 was limited to 15 real-right public training episodes and its high-motion validation "
            "error stopped improving. Add the previously audited 25 public-left episodes only through "
            "an exact physical left-to-right reflection, extend the supervised horizon to 32 frames, "
            "and select solely on real episode-disjoint right-arm validation."
        ),
        "parent": str(PARENT),
        "work_output": str(WORK / "model"),
        "data": {
            "split": str(SPLIT),
            "train_real_right_episodes": train_right,
            "train_mirrored_left_episodes": train_left,
            "validation_real_right_episodes": validation_right,
            "effective_train_right_episodes": 40,
            "outcomes_or_rewards_read": False,
            "hidden_or_final_evaluation_data": False,
        },
        "mirror_contract": {
            "image": "horizontal flip",
            "action": "swap left7/right7 and multiply each destination block by [-1,1,1,1,-1,-1,1]",
            "audit": str(MIRROR_AUDIT),
            "audit_sha256": sha256(MIRROR_AUDIT),
        },
        "fixed_training": {
            "steps_max": 100,
            "batch_size": 1,
            "rollout_horizon": 32,
            "train_rollout_steps": 16,
            "learning_rate": 2e-7,
            "validation_interval": 25,
            "validation_samples": 32,
            "high_motion_selection_weight": 0.75,
            "early_gate_after_validations": 2,
            "minimum_relative_improvement_by_step50": 0.002,
            "seed": 1575,
        },
        "selection": (
            "minimum fixed 32-sample real-right validation metric, weighted 75% high-motion and "
            "25% overall; the frozen step-0 initialization is included as the reference"
        ),
        "hard_stop": "stop at step 50 unless best metric improves at least 0.2% over frozen step 0",
        "source_sha256": {
            "trainer": sha256(TRAINER),
            "parent_model": sha256(PARENT / "model.pt"),
            "parent_manifest": sha256(PARENT / "training_manifest.json"),
            "split": sha256(SPLIT),
        },
        "guards": {
            "official_reward_model_frozen": True,
            "policy_unchanged": True,
            "no_official_batch16_outcomes": True,
            "no_hidden_or_final_data": True,
            "no_real_competition_submission": True,
            "no_rl_authorized_by_this_stage": True,
        },
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
