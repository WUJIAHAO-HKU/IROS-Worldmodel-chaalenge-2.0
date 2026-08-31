#!/usr/bin/env python3
"""Fit an action-only causal gate from public train episodes and fixed negatives."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score


CS = (0.01, 0.1, 1.0, 10.0)
THRESHOLDS = tuple(float(value) for value in np.linspace(0.1, 0.9, 17))
NEGATIVES = ("open_gripper", "static_transport", "reverse_transport")


def episode_number(path: Path) -> int:
    match = re.match(r"episode(\d+)_", path.name)
    if match is None:
        raise ValueError(path)
    return int(match.group(1))


def transform(future: np.ndarray, history: np.ndarray, name: str) -> np.ndarray:
    result = future.copy()
    anchor = history[-1, 7:13]
    if name == "open_gripper":
        result[:, 13] = 1.0
    elif name == "static_transport":
        result[:, 7:13] = anchor
    elif name == "reverse_transport":
        result[:, 7:13] = anchor - (future[:, 7:13] - anchor)
    else:
        raise ValueError(name)
    return result


def features(history: np.ndarray, future: np.ndarray) -> np.ndarray:
    anchor = history[-1, 7:13]
    relative = future[:, 7:13] - anchor
    sequence = np.concatenate((anchor[None], future[:, 7:13]), axis=0)
    delta = np.diff(sequence, axis=0)
    path = np.linalg.norm(delta, axis=1)
    net = float(np.linalg.norm(relative[-1]))
    summary = np.asarray(
        [
            path.sum(),
            net,
            net / max(float(path.sum()), 1e-8),
            path.mean(),
            path.max(),
            np.linalg.norm(np.diff(delta, axis=0), axis=1).mean(),
        ],
        dtype=np.float32,
    )
    gripper = np.concatenate((history[-1:, 13], future[:, 13])).astype(np.float32)
    return np.concatenate((relative.reshape(-1), delta.reshape(-1), gripper, summary))


def select_paths(paths: list[Path], limit: int) -> list[Path]:
    if len(paths) <= limit:
        return paths
    indices = np.linspace(0, len(paths) - 1, limit).round().astype(int)
    return [paths[int(index)] for index in indices]


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-value))


def rates(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predicted = probabilities >= threshold
    positive = labels == 1
    negative = ~positive
    return {
        "threshold": threshold,
        "tpr": float(predicted[positive].mean()),
        "tnr": float((~predicted[negative]).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--max-positive-per-episode", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing overwrite")

    split = json.loads(args.split.read_text())
    train = set(int(value) for value in split["train_episodes"])
    arm_map = {int(key): value for key, value in split["arm_by_episode"].items()}
    right_train = sorted(episode for episode in train if arm_map[episode] == "right")
    if len(right_train) != 15:
        raise RuntimeError(f"expected 15 public train right episodes, got {right_train}")

    raw_x = []
    labels = []
    groups = []
    kinds = []
    paths_used = []
    for episode in right_train:
        paths = sorted(args.windows.glob(f"episode{episode}_*.npz"))
        eligible = []
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                future = np.asarray(data["future_actions"], dtype=np.float32)
            if float(future[:, 13].mean()) <= 0.5:
                eligible.append(path)
        for path in select_paths(eligible, args.max_positive_per_episode):
            with np.load(path, allow_pickle=False) as data:
                history = np.asarray(data["history_actions"], dtype=np.float32)
                future = np.asarray(data["future_actions"], dtype=np.float32)
            raw_x.append(features(history, future))
            labels.append(1)
            groups.append(episode)
            kinds.append("expert")
            paths_used.append(path.name)
            for name in NEGATIVES:
                raw_x.append(features(history, transform(future, history, name)))
                labels.append(0)
                groups.append(episode)
                kinds.append(name)
                paths_used.append(path.name)

    x = np.asarray(raw_x, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    group = np.asarray(groups, dtype=np.int64)
    cv_rows = []
    oof_by_c = {}
    for c_value in CS:
        oof = np.full(len(y), np.nan, dtype=np.float64)
        for heldout in right_train:
            train_mask = group != heldout
            test_mask = ~train_mask
            fold_mean = x[train_mask].mean(axis=0)
            fold_scale = np.maximum(x[train_mask].std(axis=0), 1e-6)
            fold_train = (x[train_mask] - fold_mean) / fold_scale
            fold_test = (x[test_mask] - fold_mean) / fold_scale
            model = LogisticRegression(
                C=c_value,
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=1484,
            ).fit(fold_train, y[train_mask])
            oof[test_mask] = model.predict_proba(fold_test)[:, 1]
        if not np.isfinite(oof).all():
            raise RuntimeError("incomplete grouped OOF predictions")
        oof_by_c[c_value] = oof
        for threshold in THRESHOLDS:
            row = rates(y, oof, threshold)
            row["c"] = c_value
            row["min_rate"] = min(row["tpr"], row["tnr"])
            cv_rows.append(row)

    selected = max(
        cv_rows,
        key=lambda row: (
            row["min_rate"],
            row["balanced_accuracy"],
            -abs(row["threshold"] - 0.5),
            -row["c"],
        ),
    )
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-6)
    normalized = (x - mean) / scale
    final = LogisticRegression(
        C=selected["c"],
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=1484,
    ).fit(normalized, y)
    final_probability = sigmoid(normalized @ final.coef_[0] + final.intercept_[0])
    final_rates = rates(y, final_probability, selected["threshold"])
    checks = {
        "fifteen_public_train_right_episodes": len(right_train) == 15,
        "episode_grouped_cross_validation": True,
        "oof_tpr_min_0p80": selected["tpr"] >= 0.80,
        "oof_tnr_min_0p80": selected["tnr"] >= 0.80,
        "finite_parameters": bool(
            np.isfinite(final.coef_).all()
            and np.isfinite(final.intercept_).all()
            and np.isfinite(mean).all()
            and np.isfinite(scale).all()
        ),
    }
    passed = all(checks.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        feature_mean=mean.astype(np.float32),
        feature_scale=scale.astype(np.float32),
        coefficient=final.coef_[0].astype(np.float32),
        intercept=np.asarray(final.intercept_[0], dtype=np.float32),
        threshold=np.asarray(selected["threshold"], dtype=np.float32),
        c=np.asarray(selected["c"], dtype=np.float32),
        feature_version=np.asarray("v311-relative-delta-gripper-summary-v1"),
    )
    artifact_hash = hashlib.sha256(args.output.read_bytes()).hexdigest()
    report = {
        "format": "strict-track2-v311-public-action-causal-gate-training-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train_right_episodes": right_train,
        "positive_windows": int((y == 1).sum()),
        "negative_windows": int((y == 0).sum()),
        "negative_kinds": list(NEGATIVES),
        "feature_dimension": int(x.shape[1]),
        "selection": selected,
        "final_train_rates": final_rates,
        "checks": checks,
        "passed": passed,
        "artifact": str(args.output),
        "artifact_sha256": artifact_hash,
        "guards": {
            "public_train40_only": True,
            "validation_or_local_test_read": False,
            "simulator_outcomes_or_rewards_used": False,
            "policy_modified": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
