#!/usr/bin/env python3
"""Prospectively register the exact v445 full-mirror adapter and one-shot S1."""

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
    train = sorted(int(value) for value in split["train_episodes"])
    holdout = sorted(int(value) for value in split["validation_episodes"])
    if len(train) != 40 or len(holdout) != 10 or set(train) & set(holdout):
        raise RuntimeError("v445 requires immutable public train40/holdout10")
    files = {
        "runtime": ROOT / "pipeline/wam_pipeline/v445_v169_full_mirror_runtime.py",
        "s0_auditor": ROOT / "pipeline/scripts/audit_v445_full_mirror_static.py",
        "packager": ROOT / "pipeline/scripts/package_v445_full_mirror_release.py",
        "s1_generator": ROOT / "pipeline/scripts/generate_v445_s1_offline.py",
        "s1_auditor": ROOT / "pipeline/scripts/audit_v445_s1_offline.py",
        "split": args.split,
        "v169_manifest": J / "v169_instruction_arm_routed_release/v169_arm_routed_manifest.json",
    }
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    payload = {
        "format": "strict-track2-v445-full-mirror-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "classification": "frozen v169 parent diagnostic; no policy/RL authority",
        "formula": {
            "gate": "instruction contains right arm and excludes left arm",
            "context_rgb": "horizontal flip on width axis",
            "joint14": "swap left/right 7D blocks and multiply each by [-1,1,1,1,-1,-1,1]",
            "prompt": "case-insensitive all right arm occurrences -> left arm",
            "model": "single frozen v169 call with original seed forwarded unchanged",
            "output": "horizontal flip v169 prediction back to original coordinates",
            "fallback": "left/non-explicit returns same-request v169 bitexact",
        },
        "s0_gate": {
            "joint_involution": True, "rgb_involution": True, "prompt_golden": True,
            "joint_golden": True, "right_path_becomes_left_path": True,
            "left_and_nonexplicit_bitexact": True,
            "runtime_no_reward": True, "no_seed_selection": True,
            "no_action_generation": True, "all_required": True,
        },
        "s1_gate": {
            "run_count_exact": 1, "fixed_right_samples": 32, "horizon": 32,
            "first8_rgb_mae_ratio_mirror_to_v169_max": 0.998,
            "recursive32_rgb_mae_ratio_mirror_to_v169_max": 1.0,
            "recursive32_chunk3_ratio_max": 1.0, "recursive32_chunk4_ratio_max": 1.0,
            "true_to_phase_shuffle_target_mae_ratio_max": 0.998,
            "true_better_than_shuffle_sample_fraction_min": 0.60,
            "shuffle_minus_true_mae_margin_min_exclusive": 0.0,
            "true_final_reward_better_than_shuffle_fraction_min": 0.60,
            "true_minus_shuffle_final_reward_margin_min_exclusive": 0.0,
            "mixed_batch_left_same_request_bitexact": True,
            "right_serial_batch_bitexact": True,
            "right_batch_permutation_bitexact": True,
            "each_phase_first8_and_recursive32_ratio_max": 1.0,
            "each_phase_rgb_wins_min": 4,
            "improved_right_episodes_min": 3,
            "reward_counterfactuals": ["open", "static", "reverse"],
            "each_reward_counterfactual_pairwise_min": 0.75,
            "each_reward_counterfactual_normalized_margin_min": 0.05,
            "each_reward_counterfactual_rgb_effect_min_exclusive": 0.000001,
            "right_gate_fraction_exact": 1.0,
            "left_probe_bitexact": True,
            "all_required": True, "on_fail": "reject lineage; no service/RL",
            "on_pass": "diagnostic only; no automatic service/RL",
        },
        "evidence_sha256": {key: sha256(path) for key, path in files.items()},
        "guards": {
            "development_runs": 0, "reward_used_only_by_frozen_s1_binary_gate": True,
            "reward_used_by_runtime_or_selection": False, "outcome_used": False,
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
