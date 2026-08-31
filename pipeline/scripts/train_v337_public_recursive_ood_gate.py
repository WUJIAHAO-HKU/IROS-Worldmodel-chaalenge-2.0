#!/usr/bin/env python3
"""Fit a recursive-context corruption detector on public train episodes only."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from wam_pipeline.v337_public_recursive_ood_gate import FEATURE_VERSION, context_quality_features
from wam_pipeline.v326_blended_phase_terminal_runtime import Track2V326BlendedPhaseTerminal


CS = (0.01, 0.1, 1.0)
THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99)
CORRUPTION_MAE_FLOOR = 15.0


def deterministic_seed(path: Path) -> int:
    return int.from_bytes(hashlib.sha256(path.name.encode()).digest()[:8], "little") % (2**31)


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def rates(labels: np.ndarray, probabilities: np.ndarray, kinds: np.ndarray, threshold: float) -> dict:
    predicted = probabilities >= threshold
    positive = labels == 1
    negative = ~positive
    teacher = kinds == "teacher"
    return {
        "threshold": threshold,
        "corrupt_recall": float(predicted[positive].mean()),
        "negative_specificity": float((~predicted[negative]).mean()),
        "teacher_specificity": float((~predicted[teacher]).mean()),
        "balanced_accuracy": float((predicted[positive].mean() + (~predicted[negative]).mean()) / 2.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "checkpoint-dir", "library-index", "action-gate", "phase-gate",
        "windows", "split", "preregistration", "output", "report",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite v337 evidence")
    prereg = json.loads(args.preregistration.read_text())
    if prereg.get("format") != "strict-track2-v337-public-recursive-ood-preregistration-v1":
        raise RuntimeError("wrong v337 preregistration")
    if float(prereg["training_rule"]["corruption_mae_floor"]) != CORRUPTION_MAE_FLOOR:
        raise RuntimeError("corruption floor differs from preregistration")
    if tuple(prereg["selection_rule"]["c_values"]) != CS:
        raise RuntimeError("C sweep differs from preregistration")
    if tuple(prereg["selection_rule"]["thresholds"]) != THRESHOLDS:
        raise RuntimeError("threshold sweep differs from preregistration")

    split = json.loads(args.split.read_text())
    train = set(int(value) for value in split["train_episodes"])
    arm = {int(key): value for key, value in split["arm_by_episode"].items()}
    episodes = sorted(episode for episode in train if arm[episode] == "right")
    if len(episodes) != 15:
        raise RuntimeError(f"expected 15 public-train right episodes, got {episodes}")

    runtime = Track2V326BlendedPhaseTerminal(
        args.checkpoint_dir, args.library_index, args.device, args.action_gate, args.phase_gate
    )
    states = []
    for episode in episodes:
        available = {
            int(path.stem.split("_")[1]): path
            for path in args.windows.glob(f"episode{episode}_*.npz")
        }
        for alignment in range(8):
            if alignment not in available:
                continue
            with np.load(available[alignment], allow_pickle=False) as payload:
                initial = payload["context_frames"].astype(np.uint8)
            states.append({
                "episode": episode, "alignment": alignment, "start": alignment,
                "available": available, "recursive_context": initial,
            })

    features = []
    labels = []
    groups = []
    kinds = []
    errors = []
    rows = []
    while True:
        active = [state for state in states if state["start"] in state["available"]]
        if not active:
            break
        requests = []
        for state in active:
            path = state["available"][state["start"]]
            with np.load(path, allow_pickle=False) as payload:
                real = payload["context_frames"].astype(np.uint8)
                history = payload["history_actions"].astype(np.float32)
                future = payload["future_actions"].astype(np.float32)
            recursive = state["recursive_context"]
            error = float(np.abs(recursive.astype(np.int16) - real.astype(np.int16)).mean())
            teacher_feature = context_quality_features(real)
            recursive_feature = context_quality_features(recursive)
            features.append(teacher_feature); labels.append(0); groups.append(state["episode"]); kinds.append("teacher"); errors.append(0.0)
            recursive_label = int(error >= CORRUPTION_MAE_FLOOR)
            features.append(recursive_feature); labels.append(recursive_label); groups.append(state["episode"]); kinds.append("recursive"); errors.append(error)
            rows.append({
                "episode": state["episode"], "alignment": state["alignment"],
                "start": state["start"], "recursive_context_rgb_mae": error,
                "recursive_label": recursive_label,
            })
            requests.append((state, path, history, future))
        predictions = []
        for begin in range(0, len(requests), args.batch_size):
            batch = requests[begin : begin + args.batch_size]
            predictions.extend(runtime.predict_batch(
                np.stack([item[0]["recursive_context"] for item in batch]),
                np.stack([item[2] for item in batch]),
                np.stack([item[3] for item in batch]),
                np.asarray([deterministic_seed(item[1]) for item in batch], dtype=np.int64),
                ["Adjust bottle" for _ in batch],
            ))
        for (state, _, _, _), prediction in zip(requests, predictions, strict=True):
            state["recursive_context"] = prediction[-5:].copy()
            state["start"] += 8

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    group = np.asarray(groups, dtype=np.int64)
    kind = np.asarray(kinds)
    cv_rows = []
    for c_value in CS:
        oof = np.full(len(y), np.nan, dtype=np.float64)
        for heldout in episodes:
            train_mask = group != heldout
            test_mask = ~train_mask
            mean = x[train_mask].mean(axis=0)
            scale = np.maximum(x[train_mask].std(axis=0), 1e-6)
            model = LogisticRegression(
                C=c_value, max_iter=2000, class_weight="balanced",
                solver="lbfgs", random_state=1507,
            ).fit((x[train_mask] - mean) / scale, y[train_mask])
            oof[test_mask] = model.predict_proba((x[test_mask] - mean) / scale)[:, 1]
        if not np.isfinite(oof).all():
            raise RuntimeError("incomplete grouped OOF predictions")
        for threshold in THRESHOLDS:
            row = rates(y, oof, kind, threshold)
            row["c"] = c_value
            row["eligible"] = bool(
                row["teacher_specificity"] >= 0.99
                and row["negative_specificity"] >= 0.95
                and row["corrupt_recall"] >= 0.70
            )
            cv_rows.append(row)
    eligible = [row for row in cv_rows if row["eligible"]]
    selected = max(
        eligible,
        key=lambda row: (row["corrupt_recall"], row["negative_specificity"], row["teacher_specificity"], -row["c"]),
    ) if eligible else None

    checks = {
        "fifteen_public_train_right_episodes": len(episodes) == 15,
        "all_eight_alignments": len(states) == 15 * 8,
        "sufficient_corrupt_positives": int(y.sum()) >= 500,
        "episode_grouped_cross_validation": True,
        "eligible_operating_point_found": selected is not None,
    }
    passed = all(checks.values())
    artifact_hash = None
    final_rates = None
    if passed:
        mean = x.mean(axis=0)
        scale = np.maximum(x.std(axis=0), 1e-6)
        model = LogisticRegression(
            C=selected["c"], max_iter=2000, class_weight="balanced",
            solver="lbfgs", random_state=1507,
        ).fit((x - mean) / scale, y)
        probability = sigmoid(((x - mean) / scale) @ model.coef_[0] + model.intercept_[0])
        final_rates = rates(y, probability, kind, selected["threshold"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            feature_mean=mean.astype(np.float32), feature_scale=scale.astype(np.float32),
            coefficient=model.coef_[0].astype(np.float32),
            intercept=np.asarray(model.intercept_[0], dtype=np.float32),
            threshold=np.asarray(selected["threshold"], dtype=np.float32),
            c=np.asarray(selected["c"], dtype=np.float32),
            corruption_mae_floor=np.asarray(CORRUPTION_MAE_FLOOR, dtype=np.float32),
            feature_version=np.asarray(FEATURE_VERSION),
        )
        artifact_hash = hashlib.sha256(args.output.read_bytes()).hexdigest()

    report = {
        "format": "strict-track2-v337-public-recursive-ood-training-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train_episodes": episodes,
        "replay_rows": len(rows),
        "examples": len(y),
        "positive_corrupt_contexts": int(y.sum()),
        "feature_dimension": int(x.shape[1]),
        "selection": selected,
        "final_train_rates": final_rates,
        "checks": checks,
        "passed": passed,
        "artifact_sha256": artifact_hash,
        "guards": {
            "public_train_episodes_only": True,
            "labels_use_only_recursive_rgb_error_to_public_ground_truth": True,
            "rewards_or_success_outcomes_used": False,
            "public_holdout_or_evaluation_outcomes_used": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
        "cv_rows": cv_rows,
        "rows": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("replay_rows", "examples", "positive_corrupt_contexts", "selection", "checks", "passed")}, indent=2), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
