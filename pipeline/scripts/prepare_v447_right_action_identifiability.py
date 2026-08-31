#!/usr/bin/env python3
"""Preregister the train-only V447 right-action identifiability probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
SEED = 1600
EXPECTED_SPLIT_FORMAT = "strict-track2-v205-public-demo-40train-10holdout-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def make_folds(right_episodes: list[int]) -> list[list[int]]:
    ordered = sorted(
        right_episodes,
        key=lambda episode: hashlib.sha256(
            f"v447/seed{SEED}/episode{episode}".encode()
        ).hexdigest(),
    )
    return [sorted(ordered[index : index + 3]) for index in range(0, 15, 3)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    split = json.loads(args.split.read_text())
    if split.get("format") != EXPECTED_SPLIT_FORMAT:
        raise RuntimeError("V447 requires the frozen public train40/validation10 split")
    train = [int(value) for value in split["train_episodes"]]
    validation = [int(value) for value in split["validation_episodes"]]
    arms = {int(key): str(value) for key, value in split["arm_by_episode"].items()}
    right = sorted(episode for episode in train if arms[episode] == "right")
    folds = make_folds(right)
    if (
        len(train) != 40
        or len(validation) != 10
        or len(right) != 15
        or set(train) & set(validation)
        or sorted(sum(folds, [])) != right
    ):
        raise RuntimeError("V447 train40/right15/five-fold closure failed")

    files = {
        "probe": ROOT / "pipeline/scripts/probe_v447_right_action_identifiability.py",
        "auditor": ROOT / "pipeline/scripts/audit_v447_right_action_identifiability.py",
        "split": args.split,
    }
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    payload = {
        "format": "strict-track2-v447-right-action-identifiability-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "classification": "train-only data identifiability S0; no model/service/S1/RL/formal authority",
        "seed": SEED,
        "data": {
            "source": split["source"],
            "windows": split["windows"],
            "train_episodes": train,
            "validation_episodes_excluded": validation,
            "right_train_episodes": right,
            "fold_holdout_episodes": folds,
            "fold_fit_episodes_each": 12,
            "fold_holdout_episodes_each": 3,
            "close_event_windows_per_episode_exact": 8,
            "total_rows_exact": 120,
            "selection": "history-last right gripper open and first future close then held; no prompt/action-path deployment gate",
        },
        "probe": {
            "target": "first8 GT dynamic field: 16x16 block-mean(target_t - context_last), fit-fold-only PCA16 whitening",
            "context": "16x16 context_last RGB plus context_last-context_previous delta, fit-fold-only PCA32 whitening",
            "action": "history4+future8 absolute joint14, fit-fold-only standardization and PCA32 whitening",
            "phase_shuffle": "rotate future8 across held episodes with the same first-close index; retain each sample history4/context",
            "models": "same 64-column ridge design and lambda: context-only=[context32,zeros32] versus action=[context32,action32]",
            "ridge_alpha": 1.0,
            "numeric_domain": "float64 regression and latent error; no uint8 rounding, residual cap, runtime gate, v169, reward, or outcome",
        },
        "identifiability_gate": {
            "action_error_over_context_error_max": 0.95,
            "action_error_over_phase_shuffle_error_max": 0.97,
            "passing_folds_min": 4,
            "passing_episodes_min": 12,
            "all_required": True,
            "on_pass": "authorize one separately preregistered train-only first8 dynamics-latent RGB candidate; never S1/RL/formal",
            "on_fail": "declare current right15 insufficient for action-conditioned first8 dynamics and require paired public-simulator interventions before another model",
        },
        "descriptive_only": [
            "per-fold and global action numerical rank and 95-percent effective rank",
            "exact same-context/different-action target groups and pairs",
            "near-context/same-phase/different-action support",
            "failure-labelled target and intervention/counterfactual target counts",
        ],
        "resources": {
            "device": "CPU only; CUDA_VISIBLE_DEVICES empty",
            "cpu_affinity": "0-11",
            "blas_threads": 6,
            "estimated_peak_ram_gib_max": 2,
            "estimated_new_disk_mib_max": 50,
            "gpu_vram_mib": 0,
        },
        "evidence_sha256": {key: sha256(path) for key, path in files.items()},
        "guards": {
            "public_train_only": True,
            "validation_development_or_final_used": False,
            "reward_used": False,
            "outcome_used_for_fit_or_gate": False,
            "outcome_labels_counted_descriptive_only": True,
            "policy_loaded_or_modified": False,
            "policy_updates": 0,
            "hidden_or_final_data": False,
            "real_submission": False,
            "s1_authorized": False,
            "rl_authorized": False,
            "formal_authorized": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
