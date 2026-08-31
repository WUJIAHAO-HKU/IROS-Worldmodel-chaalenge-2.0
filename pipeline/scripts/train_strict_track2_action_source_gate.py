#!/usr/bin/env python3
"""Train an episode-disjoint request-action router for Track 2 parent experts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ACTION_DIM = 14
HISTORY = 4
FUTURE = 8


def action_features(history: np.ndarray, future: np.ndarray) -> np.ndarray:
    actions = np.concatenate((history, future), axis=0).astype(np.float32)
    if actions.shape != (HISTORY + FUTURE, ACTION_DIM):
        raise ValueError(f"unexpected action sequence {actions.shape}")
    delta = np.diff(actions, axis=0)
    summary = np.concatenate((
        actions.mean(0), actions.std(0),
        np.abs(delta[:, :7]).mean(0), np.abs(delta[:, 7:]).mean(0),
        np.asarray([np.abs(delta[:, :7]).mean(), np.abs(delta[:, 7:]).mean()], dtype=np.float32),
    ))
    return np.concatenate((actions.reshape(-1), delta.reshape(-1), summary)).astype(np.float32)


def select(records: list[dict], split: str, maximum: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    selected = []
    for source in ("official", "synthetic"):
        rows = [
            row for row in records
            if row["split"] == split and row["source"] == source and int(row.get("replica", 0)) == 0
        ]
        order = rng.permutation(len(rows))[:min(maximum, len(rows))]
        selected.extend(rows[index] for index in order)
    return selected


def load(root: Path, records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    features, labels = [], []
    for record in records:
        with np.load(root / record["path"], allow_pickle=False) as values:
            features.append(action_features(values["history_actions"], values["future_actions"]))
        labels.append(record["source"] == "synthetic")
    return np.stack(features), np.asarray(labels, dtype=np.float32)


def metrics(probability: np.ndarray, labels: np.ndarray) -> dict:
    result = {}
    accuracies = []
    for name, label in (("official", 0), ("onpolicy", 1)):
        values = probability[labels == label]
        accuracy = float(((values >= .5) == bool(label)).mean())
        accuracies.append(accuracy)
        result[name] = {
            "windows": int(len(values)),
            "accuracy": accuracy,
            "mean_probability_onpolicy": float(values.mean()),
            "probability_quantiles": [float(v) for v in np.quantile(values, (0, .01, .1, .5, .9, .99, 1))],
        }
    result["balanced_accuracy"] = float(np.mean(accuracies))
    return result


class Gate(torch.nn.Module):
    def __init__(self, features: int, hidden: int) -> None:
        super().__init__()
        self.first = torch.nn.Linear(features, hidden)
        self.second = torch.nn.Linear(hidden, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.second(F.silu(self.first(value)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-train-per-source", type=int, default=3000)
    args = parser.parse_args()

    root = Path(args.windows).resolve()
    manifest = Path(args.source_manifest).resolve()
    preregistration = Path(args.preregistration).resolve()
    prereg = json.loads(preregistration.read_text())
    config = prereg["classifier"]
    records = json.loads(manifest.read_text())
    train_records = select(records, "train", args.max_train_per_source, int(config["seed"]))
    validation_records = select(records, "validation", 1_000_000, int(config["seed"]) + 1)
    train_x, train_y = load(root, train_records)
    validation_x, validation_y = load(root, validation_records)
    mean, std = train_x.mean(0), train_x.std(0) + 1e-6
    train_x = (train_x - mean) / std
    validation_x = (validation_x - mean) / std

    torch.manual_seed(int(config["seed"]))
    model = Gate(train_x.shape[1], int(config["hidden_features"]))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    inputs = torch.from_numpy(train_x)
    labels = torch.from_numpy(train_y)[:, None]
    positive = max(1, int(train_y.sum()))
    negative = max(1, int((train_y == 0).sum()))
    weights = torch.from_numpy(np.where(
        train_y == 1, len(train_y) / (2 * positive), len(train_y) / (2 * negative)
    ).astype(np.float32))[:, None]
    for _ in range(int(config["steps"])):
        optimizer.zero_grad(set_to_none=True)
        loss = (F.binary_cross_entropy_with_logits(model(inputs), labels, reduction="none") * weights).mean()
        loss.backward()
        optimizer.step()

    with torch.inference_mode():
        train_probability = model(inputs).sigmoid().squeeze(1).numpy()
        validation_probability = model(torch.from_numpy(validation_x)).sigmoid().squeeze(1).numpy()
    train_metrics = metrics(train_probability, train_y)
    validation_metrics = metrics(validation_probability, validation_y)
    gates = prereg["admission_gates"]
    passed = (
        validation_metrics["balanced_accuracy"] >= float(gates["heldout_balanced_accuracy"])
        and validation_metrics["official"]["accuracy"] >= float(gates["heldout_official_accuracy"])
        and validation_metrics["onpolicy"]["accuracy"] >= float(gates["heldout_onpolicy_accuracy"])
    )

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    checkpoint = {
        "format": "strict-track2-action-source-gate-v1",
        "history_actions": HISTORY,
        "future_actions": FUTURE,
        "action_dim": ACTION_DIM,
        "feature_mean": torch.from_numpy(mean),
        "feature_std": torch.from_numpy(std),
        "hidden_features": int(config["hidden_features"]),
        "state_dict": model.state_dict(),
        "threshold": .5,
    }
    torch.save(checkpoint, output / "action_source_gate.pt")
    report = {
        "format": checkpoint["format"],
        "passed": passed,
        "windows": str(root),
        "source_manifest": str(manifest),
        "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "preregistration": str(preregistration),
        "preregistration_sha256": hashlib.sha256(preregistration.read_bytes()).hexdigest(),
        "feature_count": int(train_x.shape[1]),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train": train_metrics,
        "validation": validation_metrics,
    }
    (output / "training_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit("action source gate failed preregistered held-out accuracy")


if __name__ == "__main__":
    main()
