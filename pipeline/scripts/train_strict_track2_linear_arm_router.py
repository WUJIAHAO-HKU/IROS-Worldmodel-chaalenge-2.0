#!/usr/bin/env python3
"""Fit a tiny arm router on train episodes and audit held-out episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from wam_pipeline.arm_router import FORMAT, arm_router_features


def episode(path: Path) -> int:
    match = re.match(r"episode(\d+)_", path.name)
    if not match:
        raise ValueError(f"invalid window name: {path.name}")
    return int(match.group(1))


def confusion(target: np.ndarray, predicted: np.ndarray) -> list[list[int]]:
    return [
        [int(np.count_nonzero((target == truth) & (predicted == guess))) for guess in (0, 1)]
        for truth in (0, 1)
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=12)
    parser.add_argument("--c", type=float, default=1.0)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite arm-router release: {args.output}")
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    if split.get("real_acceptance_seeds_used") is not False:
        raise SystemExit("split must explicitly exclude real acceptance seeds")
    train_episodes = set(int(value) for value in split["train_episodes"])
    validation_episodes = set(int(value) for value in split["validation_episodes"])
    if train_episodes & validation_episodes:
        raise SystemExit("train and validation episodes overlap")

    features, targets, episodes = [], [], []
    for path in sorted(args.windows.glob("episode*_*.npz")):
        current_episode = episode(path)
        if current_episode not in train_episodes | validation_episodes:
            continue
        with np.load(path, allow_pickle=False) as values:
            if "arm_right" not in values.files:
                continue
            features.append(
                arm_router_features(
                    values["context_frames"],
                    values["history_actions"],
                    values["future_actions"],
                    args.image_size,
                )
            )
            targets.append(bool(values["arm_right"]))
            episodes.append(current_episode)
    x = np.stack(features)
    y = np.asarray(targets, dtype=np.int64)
    episode_ids = np.asarray(episodes, dtype=np.int64)
    train = np.isin(episode_ids, list(train_episodes))
    validation = np.isin(episode_ids, list(validation_episodes))
    if not train.any() or not validation.any():
        raise SystemExit("empty train or validation partition")

    scaler = StandardScaler().fit(x[train])
    classifier = LogisticRegression(
        C=args.c, max_iter=1000, class_weight="balanced", random_state=169
    ).fit(scaler.transform(x[train]), y[train])
    prediction = classifier.predict(scaler.transform(x[validation]))
    raw_prediction = []
    for row in x[validation]:
        action_offset = 3 * args.image_size * args.image_size
        # The legacy router compares the mean absolute delta of each 7-D arm.
        # In this feature layout those are action block number six.
        delta_absolute = row[action_offset + 5 * 14 : action_offset + 6 * 14]
        raw_prediction.append(int(delta_absolute[7:].mean() > delta_absolute[:7].mean()))
    raw_prediction = np.asarray(raw_prediction, dtype=np.int64)
    validation_y = y[validation]
    validation_episode_ids = episode_ids[validation]
    majority = []
    for current_episode in sorted(set(validation_episode_ids.tolist())):
        mask = validation_episode_ids == current_episode
        majority.append(
            {
                "episode": current_episode,
                "target_right": bool(validation_y[mask][0]),
                "predicted_right": bool(prediction[mask].mean() >= 0.5),
                "window_accuracy": float((prediction[mask] == validation_y[mask]).mean()),
            }
        )

    args.output.mkdir(parents=True)
    model_path = args.output / "arm_router.npz"
    np.savez_compressed(
        model_path,
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
        weight=classifier.coef_[0].astype(np.float32),
        bias=np.asarray(classifier.intercept_[0], dtype=np.float32),
    )
    manifest = {
        "format": FORMAT,
        "classification": "world-model expert routing only; never changes or selects policy actions",
        "image_size": args.image_size,
        "feature_count": int(x.shape[1]),
        "logistic_c": args.c,
        "model_sha256": sha256(model_path),
        "data": {
            "windows": str(args.windows.resolve()),
            "split_manifest": str(args.split_manifest.resolve()),
            "split_manifest_sha256": sha256(args.split_manifest),
            "train_windows": int(train.sum()),
            "validation_windows": int(validation.sum()),
            "train_episodes": len(set(episode_ids[train].tolist())),
            "validation_episodes": len(set(validation_episode_ids.tolist())),
            "real_acceptance_seeds_used": False,
        },
        "validation": {
            "learned_window_accuracy": float((prediction == validation_y).mean()),
            "legacy_window_accuracy": float((raw_prediction == validation_y).mean()),
            "absolute_accuracy_gain_points": float(
                100.0 * ((prediction == validation_y).mean() - (raw_prediction == validation_y).mean())
            ),
            "confusion": confusion(validation_y, prediction),
            "legacy_confusion": confusion(validation_y, raw_prediction),
            "episode_majority_accuracy": float(
                np.mean([row["predicted_right"] == row["target_right"] for row in majority])
            ),
            "episodes": majority,
        },
        "selection_disclosure": "C=1 linear model chosen after a deployability-focused exploratory comparison on this validation split; real RoboTwin seeds remain untouched acceptance data.",
    }
    (args.output / "arm_router_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
