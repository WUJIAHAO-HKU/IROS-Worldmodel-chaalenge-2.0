#!/usr/bin/env python3
"""Prospectively register v444 train12/kill3 direct-residual step25."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
SEED = 1595


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def episode_partition(right: list[int]) -> tuple[list[int], list[int]]:
    ordered = sorted(right, key=lambda episode: hashlib.sha256(f"v444/seed{SEED}/episode{episode}".encode()).hexdigest())
    return sorted(ordered[:12]), sorted(ordered[12:])


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
        raise RuntimeError("v444 requires immutable public train40/right15")
    fit, kill = episode_partition(right)
    files = {
        "trainer": ROOT / "pipeline/scripts/train_v444_direct_residual_step25.py",
        "runtime": ROOT / "pipeline/wam_pipeline/v444_v169_direct_residual_runtime.py",
        "s0_auditor": ROOT / "pipeline/scripts/audit_v444_step25_killgate.py",
        "packager": ROOT / "pipeline/scripts/package_v444_direct_residual_release.py",
        "split": args.split,
        "v169_manifest": J / "v169_instruction_arm_routed_release/v169_arm_routed_manifest.json",
    }
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    payload = {
        "format": "strict-track2-v444-direct-residual-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "classification": "parent world-model diagnostic only; no policy/RL authority",
        "seed": SEED,
        "data": {
            "public_train_episodes": train, "validation_episodes_excluded": validation,
            "right_train_episodes": right, "fit_episodes": fit, "kill_episodes": kill,
            "windows_per_episode_exact": 8, "total_windows_exact": 120,
            "fit_windows_exact": 96, "kill_windows_exact": 24,
            "partition": "sort SHA256(v444/seed1595/episodeN); first12 fit, final3 kill",
            "selection": "v442 close-only action gate; no target/reward/outcome in selection",
        },
        "model": {
            "baseline": "frozen original v169 RGB",
            "supervision": "clip(public-train target RGB - v169 RGB, -8, 8)",
            "architecture": "two 16-channel RGB convs; explicit joint12x14 plus right/left arm-token FiLM; zero-init output conv",
            "runtime_gate": "explicit-right + right7D path>left7D + history-last-open + future-close-then-held",
            "protected_frames_exact_zero": [0, 1, 6, 7],
            "residual_abs_max": 8.0,
        },
        "training": {
            "steps": 25, "batch_size": 4, "optimizer": "AdamW",
            "learning_rate": 0.001, "weight_decay": 0.0001,
            "loss": "mean absolute residual error on unprotected frames",
            "initialization": "all standard except final conv exact zero",
            "checkpoint_count": 1, "inference_batch_size": 4,
        },
        "kill_gate": {
            "kill_episodes_exact": 3, "kill_windows_exact": 24,
            "residual_mae_ratio_step25_to_init_max": 0.99,
            "rgb_mae_ratio_step25_to_init_max": 0.99,
            "kill_episode_rgb_improved_exact": 3,
            "recursive32_rgb_mae_ratio_step25_to_init_max": 1.0,
            "recursive32_chunk3_rgb_mae_ratio_step25_to_init_max": 1.0,
            "recursive32_chunk4_rgb_mae_ratio_step25_to_init_max": 1.0,
            "protected_frames_bitexact": True, "left_and_g0_bitexact": True,
            "candidate_minus_v169_pixel_abs_max": 8,
            "all_required": True, "on_fail": "retain forensic checkpoint/report; no package/S1/RL",
            "on_pass": "package parent diagnostic only; S1 requires separate preregistration",
        },
        "evidence_sha256": {key: sha256(path) for key, path in files.items()},
        "guards": {
            "development_or_final_used": False, "reward_or_outcome_used": False,
            "official_pi05_modified": False, "official_reward_modified": False,
            "policy_updates": 0, "real_submission": False, "rl_authorized": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
