#!/usr/bin/env python3
"""Prospectively register v446 five-fold train-only contrastive residual training."""

from __future__ import annotations

import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from wam_pipeline.v442_v169_close_aligned_projection_runtime import gate_decision


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
SEED = 1597


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""): h.update(block)
    return h.hexdigest()


def folds(right: list[int]) -> list[list[int]]:
    ordered = sorted(right, key=lambda episode: hashlib.sha256(f"v446/seed{SEED}/episode{episode}".encode()).hexdigest())
    return [sorted(ordered[index:index + 3]) for index in range(0, 15, 3)]


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def selected_source_manifest(windows: Path, right: list[int], prompts: dict[int, str]) -> dict:
    selected: dict[str, str] = {}
    counts = {episode: 0 for episode in right}
    for episode in right:
        for path in sorted(windows.glob(f"episode{episode}_*.npz")):
            with np.load(path, allow_pickle=False) as data:
                history = np.asarray(data["history_actions"], dtype=np.float32)
                future = np.asarray(data["future_actions"], dtype=np.float32)
            decision = gate_decision(history, future, prompts[episode])
            if decision.get("gate") and decision.get("phase") == "close":
                selected[path.name] = sha256(path); counts[episode] += 1
    if len(selected) != 120 or any(value != 8 for value in counts.values()):
        raise RuntimeError(f"v446 source manifest close120 drift: {counts}")
    body = {"format": "strict-track2-v446-selected-window-source-manifest-v1", "files": selected}
    return {**body, "canonical_sha256": canonical_sha(body)}


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("split", "windows", "v169-release-manifest", "v169-library-manifest", "v169-library-split-manifest", "reward-checkpoint", "t5-config", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    split = json.loads(args.split.read_text())
    arms = {int(key): str(value) for key, value in split["arm_by_episode"].items()}
    train = [int(value) for value in split["train_episodes"]]; validation = [int(value) for value in split["validation_episodes"]]
    right = sorted(episode for episode in train if arms[episode] == "right")
    prompts = {int(key): str(value) for key, value in split["episode_to_instruction"].items()}
    fold_rows = folds(right)
    if len(train) != 40 or len(validation) != 10 or len(right) != 15 or sorted(sum(fold_rows, [])) != right or set(train) & set(validation):
        raise RuntimeError("v446 immutable train40/right15/five-fold contract failed")
    files = {
        "trainer": ROOT / "pipeline/scripts/train_v446_contrastive_residual_5fold.py",
        "runtime": ROOT / "pipeline/wam_pipeline/v446_v169_contrastive_residual_unet_runtime.py",
        "auditor": ROOT / "pipeline/scripts/audit_v446_s0_contract.py",
        "packager": ROOT / "pipeline/scripts/package_v446_contrastive_residual_release.py",
        "v169_runtime": ROOT / "pipeline/wam_pipeline/v169_arm_routed_runtime.py",
        "close_gate_runtime": ROOT / "pipeline/wam_pipeline/v442_v169_close_aligned_projection_runtime.py",
        "split": args.split,
        "v169_manifest": args.v169_release_manifest,
        "v169_library_manifest": args.v169_library_manifest,
        "v169_library_split_manifest": args.v169_library_split_manifest,
        "reward_checkpoint": args.reward_checkpoint,
        "t5_config": args.t5_config,
    }
    for path in files.values():
        if not path.is_file(): raise FileNotFoundError(path)
    evidence_sha256 = {key: sha256(path) for key, path in files.items()}
    source_manifest = selected_source_manifest(args.windows.resolve(), right, prompts)
    input_contract = {}
    for contract_key, evidence_key, path in (
        ("split", "split", args.split),
        ("v169_release_manifest", "v169_manifest", args.v169_release_manifest),
        ("v169_library_manifest", "v169_library_manifest", args.v169_library_manifest),
        ("v169_library_split_manifest", "v169_library_split_manifest", args.v169_library_split_manifest),
        ("reward_checkpoint", "reward_checkpoint", args.reward_checkpoint),
        ("t5_config", "t5_config", args.t5_config),
    ):
        input_contract[contract_key] = {"resolved_path": str(path.resolve()), "sha256": evidence_sha256[evidence_key]}
    input_contract["windows"] = {"resolved_path": str(args.windows.resolve()), "selected_source_manifest_sha256": source_manifest["canonical_sha256"]}
    payload = {
        "format": "strict-track2-v446-contrastive-residual-5fold-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(), "seed": SEED,
        "classification": "train-only parent diagnostic; no S1/policy/RL authority",
        "data": {
            "train_episodes": train, "validation_episodes_excluded": validation, "right_train_episodes": right,
            "close_windows_per_episode_exact": 8, "total_close_windows_exact": 120,
            "fold_holdout_episodes": fold_rows, "fold_train_episodes_each": 12, "fold_holdout_episodes_each": 3,
            "action_variants": ["true", "shuffle", "open", "static", "reverse"],
            "baseline_cache": "frozen v169 independently queried for every action variant",
            "selected_window_source_manifest": source_manifest,
        },
        "inputs": input_contract,
        "model": {
            "architecture": "128-resolution 16/32-channel residual U-Net with explicit history4/future8+arm-token FiLM",
            "baseline": "frozen original v169", "residual_cap": 4.0, "protected_frames": [],
            "runtime_gate": "v442 close-only; left/nonexplicit/nonclose returns v169 bitexact",
        },
        "training": {
            "folds": 5, "steps_per_fold": 50, "final_steps": 50, "batch_size": 4,
            "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001,
            "loss": {"true_l1": 1.0, "negative_zero_mean": 0.25, "target_mae_hinge": 0.25, "hinge_margin_pixels_at128": 0.10},
            "negative_residual_target": 0.0, "true_target": "clip(target-v169_true,-4,4)",
            "reward_in_training": False, "developer_hyperparameter_sweep": False,
        },
        "fold_gate": {
            "each_fold_true_rgb_ratio_to_v169_max": 1.0,
            "each_fold_true_better_than_each_negative": True,
            "each_fold_each_negative_pairwise_fraction_min": 0.75,
            "each_fold_each_negative_episode_improved_exact": 3,
            "each_fold_holdout_episode_improved_exact": 3,
            "five_of_five_required": True,
            "execution_on_fold_failure": "complete all five preregistered folds and global forensic audit; never start all15",
        },
        "global_gate": {
            "true_rgb_ratio_to_v169_max": 0.992, "improved_episode_min": 12,
            "each_negative_rgb_pairwise_fraction_min": 0.75, "each_negative_rgb_margin_min_exclusive": 0.0,
            "each_negative_final_reward_pairwise_fraction_min": 0.75, "each_negative_normalized_reward_margin_min": 0.05,
            "recursive32_ratio_max": 1.0, "recursive32_chunk3_ratio_max": 1.0, "recursive32_chunk4_ratio_max": 1.0,
            "all_required": True, "on_fail": "retain forensic folds/report; no final/package/S1/RL",
        },
        "evidence_sha256": evidence_sha256,
        "guards": {
            "development_or_final_used": False, "outcome_used": False,
            "official_reward_used_only_after_frozen_fold_outputs": True,
            "official_pi05_modified": False, "official_reward_modified": False,
            "policy_updates": 0, "real_submission": False, "s1_authorized": False, "rl_authorized": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2) + "\n"); print(args.output); return 0


if __name__ == "__main__": raise SystemExit(main())
