#!/usr/bin/env python3
"""Train an episode-disjoint visual router for strict Track 2 parent experts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


POOL_SIZE = 16


def frame_features(frame: np.ndarray) -> np.ndarray:
    value = torch.from_numpy(frame).permute(2, 0, 1).float().div(255).unsqueeze(0)
    pooled = F.adaptive_avg_pool2d(value, (POOL_SIZE, POOL_SIZE)).flatten(1)
    mean = value.mean((2, 3))
    std = value.std((2, 3), unbiased=False)
    return torch.cat((pooled, mean, std), 1).squeeze(0).numpy()


def load_features(root: Path, records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    features, labels = [], []
    for record in records:
        with np.load(root / record["path"], allow_pickle=False) as values:
            frame = values["context_frames"][-1].copy()
        features.append(frame_features(frame))
        labels.append(record["source"] == "synthetic")
    return np.stack(features).astype(np.float32), np.asarray(labels, dtype=np.float32)


def select_records(records: list[dict], split: str, max_per_source: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    selected = []
    for source in ("official", "synthetic"):
        rows = [
            item for item in records
            if item["split"] == split and item["source"] == source and int(item.get("replica", 0)) == 0
        ]
        order = rng.permutation(len(rows))[: min(max_per_source, len(rows))]
        selected.extend(rows[index] for index in order)
    return selected


def source_metrics(probability: np.ndarray, labels: np.ndarray) -> dict:
    output = {}
    for name, label in (("official", 0), ("synthetic", 1)):
        values = probability[labels == label]
        output[name] = {
            "windows": int(len(values)),
            "mean_probability_synthetic": float(values.mean()),
            "probability_synthetic_quantiles": [
                float(item) for item in np.quantile(values, (0, .1, .5, .9, 1))
            ],
            "accuracy_at_0_5": float(((values >= .5) == bool(label)).mean()),
        }
    output["balanced_accuracy_at_0_5"] = float(
        .5 * (((probability[labels == 0] < .5).mean()) + ((probability[labels == 1] >= .5).mean()))
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=.02)
    parser.add_argument("--max-train-per-source", type=int, default=500)
    parser.add_argument("--max-validation-per-source", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    if min(args.steps, args.learning_rate, args.max_train_per_source, args.max_validation_per_source) <= 0:
        raise ValueError("training arguments must be positive")

    torch.manual_seed(args.seed)
    root, manifest_path = Path(args.windows), Path(args.source_manifest)
    records = json.loads(manifest_path.read_text())
    train_records = select_records(records, "train", args.max_train_per_source, args.seed)
    validation_records = select_records(records, "validation", args.max_validation_per_source, args.seed + 1)
    train_x, train_y = load_features(root, train_records)
    validation_x, validation_y = load_features(root, validation_records)
    mean, std = train_x.mean(0), train_x.std(0) + 1e-5
    train_x = (train_x - mean) / std
    validation_x = (validation_x - mean) / std

    features = torch.from_numpy(train_x)
    labels = torch.from_numpy(train_y)[:, None]
    model = torch.nn.Linear(features.shape[1], 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    positive, negative = int(train_y.sum()), int((train_y == 0).sum())
    sample_weights = torch.from_numpy(
        np.where(train_y == 1, len(train_y) / (2 * positive), len(train_y) / (2 * negative)).astype(np.float32)
    )[:, None]
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = (
            F.binary_cross_entropy_with_logits(model(features), labels, reduction="none") * sample_weights
        ).mean()
        loss.backward()
        optimizer.step()

    with torch.inference_mode():
        train_probability = model(features).sigmoid().squeeze(1).numpy()
        validation_probability = model(torch.from_numpy(validation_x)).sigmoid().squeeze(1).numpy()
    train_metrics = source_metrics(train_probability, train_y)
    validation_metrics = source_metrics(validation_probability, validation_y)
    if validation_metrics["balanced_accuracy_at_0_5"] < .99:
        raise RuntimeError("source router failed the preregistered 99% held-out balanced-accuracy gate")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": "strict-track2-visual-source-gate-v1",
        "pool_size": POOL_SIZE,
        "feature_mean": torch.from_numpy(mean),
        "feature_std": torch.from_numpy(std),
        "weight": model.weight.detach().cpu(),
        "bias": model.bias.detach().cpu(),
        "threshold": .5,
    }
    torch.save(checkpoint, output / "source_gate.pt")
    report = {
        "format": checkpoint["format"],
        "purpose": "route familiar supplied-demonstration states to frozen V15 and on-policy states to the adapted parent",
        "windows": str(root.resolve()),
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "episode_disjoint_split": True,
        "replicas_excluded": True,
        "feature_definition": "last real context RGB, adaptive 16x16 channel means, global channel mean/std",
        "parameters": int(model.weight.numel() + model.bias.numel()),
        "steps": args.steps,
        "seed": args.seed,
        "train": train_metrics,
        "validation": validation_metrics,
    }
    (output / "training_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
