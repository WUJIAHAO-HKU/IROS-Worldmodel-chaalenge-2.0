#!/usr/bin/env python3
"""Train the V15.7 router on official, Pi0.5, and hybrid reset windows."""

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
SOURCES = ("official", "onpolicy", "hybrid")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def action_features(history: np.ndarray, future: np.ndarray) -> np.ndarray:
    actions = np.concatenate((history, future), axis=0).astype(np.float32)
    if actions.shape != (HISTORY + FUTURE, ACTION_DIM):
        raise ValueError(f"unexpected action sequence {actions.shape}")
    delta = np.diff(actions, axis=0)
    summary = np.concatenate(
        (
            actions.mean(0),
            actions.std(0),
            np.abs(delta[:, :7]).mean(0),
            np.abs(delta[:, 7:]).mean(0),
            np.asarray(
                [np.abs(delta[:, :7]).mean(), np.abs(delta[:, 7:]).mean()],
                dtype=np.float32,
            ),
        )
    )
    return np.concatenate((actions.reshape(-1), delta.reshape(-1), summary)).astype(
        np.float32
    )


def select(
    records: list[dict], source: str, split: str, maximum: int, seed: int
) -> list[dict]:
    rows = [
        record
        for record in records
        if record["source"] == source
        and record["split"] == split
        and int(record.get("replica", 0)) == 0
    ]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))[: min(maximum, len(rows))]
    return [rows[int(index)] for index in order]


def load_actions(root: Path, record: dict) -> tuple[np.ndarray, np.ndarray]:
    with np.load(root / record["path"], allow_pickle=False) as values:
        return values["history_actions"].copy(), values["future_actions"].copy()


def pair_same_arm(
    official: list[dict], onpolicy: list[dict], seed: int
) -> list[tuple[dict, dict]]:
    rng = np.random.default_rng(seed)
    pairs = []
    for arm in ("left", "right"):
        official_arm = [record for record in official if record["arm"] == arm]
        onpolicy_arm = [record for record in onpolicy if record["arm"] == arm]
        if not official_arm or not onpolicy_arm:
            raise RuntimeError(f"empty hybrid stratum for {arm}")
        order = rng.permutation(len(onpolicy_arm))
        for index, official_record in enumerate(official_arm):
            pairs.append(
                (official_record, onpolicy_arm[int(order[index % len(order)])])
            )
    rng.shuffle(pairs)
    return pairs


def build_dataset(
    root: Path,
    official: list[dict],
    onpolicy: list[dict],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    features: list[np.ndarray] = []
    labels: list[float] = []
    strata: list[str] = []
    arm_counts = {source: {"left": 0, "right": 0} for source in SOURCES}
    for source, rows, label in (
        ("official", official, 0.0),
        ("onpolicy", onpolicy, 1.0),
    ):
        for record in rows:
            history, future = load_actions(root, record)
            features.append(action_features(history, future))
            labels.append(label)
            strata.append(source)
            arm_counts[source][record["arm"]] += 1

    pairs = pair_same_arm(official, onpolicy, seed)
    for official_record, onpolicy_record in pairs:
        official_history, official_future = load_actions(root, official_record)
        _, onpolicy_future = load_actions(root, onpolicy_record)
        # Preserve the Pi0.5 motion profile while anchoring it at the absolute
        # a4 state belonging to the official o0..o4 reset context.
        hybrid_future = official_future[:1] + (
            onpolicy_future - onpolicy_future[:1]
        )
        features.append(action_features(official_history, hybrid_future))
        labels.append(1.0)
        strata.append("hybrid")
        arm_counts["hybrid"][official_record["arm"]] += 1
    return (
        np.stack(features),
        np.asarray(labels, dtype=np.float32),
        np.asarray(strata),
        arm_counts,
    )


def stratum_metrics(
    probabilities: np.ndarray, strata: np.ndarray
) -> dict[str, dict]:
    result = {}
    for source in SOURCES:
        values = probabilities[strata == source]
        target = source != "official"
        result[source] = {
            "windows": int(len(values)),
            "accuracy": float(((values >= 0.5) == target).mean()),
            "mean_probability_candidate": float(values.mean()),
            "probability_quantiles": [
                float(value)
                for value in np.quantile(values, (0, 0.01, 0.1, 0.5, 0.9, 0.99, 1))
            ],
        }
    return result


class Gate(torch.nn.Module):
    def __init__(self, features: int, hidden: int) -> None:
        super().__init__()
        self.first = torch.nn.Linear(features, hidden)
        self.second = torch.nn.Linear(hidden, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.second(F.silu(self.first(value)))


def probe_metrics(
    model: Gate,
    mean: np.ndarray,
    std: np.ndarray,
    paths: list[Path],
) -> list[dict]:
    rows = []
    with torch.inference_mode():
        for path in paths:
            with np.load(path, allow_pickle=False) as values:
                histories = values["history_actions"]
                futures = values["future_actions"]
            for index, (history, future) in enumerate(zip(histories, futures)):
                feature = (action_features(history, future) - mean) / std
                probability = float(
                    model(torch.from_numpy(feature).float().unsqueeze(0))
                    .sigmoid()[0, 0]
                )
                rows.append(
                    {
                        "audit": str(path),
                        "batch_index": index,
                        "probability_candidate": probability,
                        "route": "candidate" if probability >= 0.5 else "baseline",
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--probe-audit", action="append", default=[], type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    root = args.windows.resolve()
    source_manifest = args.source_manifest.resolve()
    preregistration = args.preregistration.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    prereg = json.loads(preregistration.read_text(encoding="utf-8"))
    if prereg.get("format") != "strict-track2-v157-hybrid-action-gate-preregistration-v1":
        raise RuntimeError("unexpected V15.7 hybrid gate preregistration")
    config = prereg["classifier"]
    records = json.loads(source_manifest.read_text(encoding="utf-8"))
    maximum = int(config["max_train_per_source"])
    seed = int(config["seed"])
    train_official = select(records, "official", "train", maximum, seed)
    train_onpolicy = select(records, "synthetic", "train", maximum, seed + 1)
    validation_official = select(records, "official", "validation", 1_000_000, seed + 2)
    validation_onpolicy = select(records, "synthetic", "validation", 1_000_000, seed + 3)
    train_x, train_y, train_strata, train_arm_counts = build_dataset(
        root, train_official, train_onpolicy, seed + 4
    )
    validation_x, validation_y, validation_strata, validation_arm_counts = build_dataset(
        root, validation_official, validation_onpolicy, seed + 5
    )
    mean = train_x.mean(0)
    std = train_x.std(0) + 1e-6
    normalized_train = (train_x - mean) / std
    normalized_validation = (validation_x - mean) / std

    torch.manual_seed(seed)
    model = Gate(train_x.shape[1], int(config["hidden_features"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    inputs = torch.from_numpy(normalized_train).float()
    labels = torch.from_numpy(train_y).float()[:, None]
    # Equalize all three semantic strata, not merely binary classes.
    counts = {source: max(1, int((train_strata == source).sum())) for source in SOURCES}
    sample_weight = np.asarray(
        [len(train_strata) / (len(SOURCES) * counts[source]) for source in train_strata],
        dtype=np.float32,
    )
    weights = torch.from_numpy(sample_weight)[:, None]
    loss_value = 0.0
    for _ in range(int(config["steps"])):
        optimizer.zero_grad(set_to_none=True)
        loss = (
            F.binary_cross_entropy_with_logits(
                model(inputs), labels, reduction="none"
            )
            * weights
        ).mean()
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())

    with torch.inference_mode():
        train_probability = model(inputs).sigmoid().squeeze(1).numpy()
        validation_probability = (
            model(torch.from_numpy(normalized_validation).float())
            .sigmoid()
            .squeeze(1)
            .numpy()
        )
    train_metrics = stratum_metrics(train_probability, train_strata)
    validation_metrics = stratum_metrics(validation_probability, validation_strata)
    probe_paths = [path.resolve() for path in args.probe_audit]
    probes = probe_metrics(model, mean, std, probe_paths)
    probe_candidate_fraction = float(
        np.mean([row["route"] == "candidate" for row in probes])
    ) if probes else 0.0
    gates = prereg["admission_gates"]
    gate_results = {
        "heldout_official_accuracy": validation_metrics["official"]["accuracy"]
        >= float(gates["heldout_official_accuracy_min"]),
        "heldout_onpolicy_accuracy": validation_metrics["onpolicy"]["accuracy"]
        >= float(gates["heldout_onpolicy_accuracy_min"]),
        "heldout_hybrid_accuracy": validation_metrics["hybrid"]["accuracy"]
        >= float(gates["heldout_hybrid_accuracy_min"]),
        "corrected_rollout_probes": probe_candidate_fraction
        >= float(gates["corrected_rollout_probe_candidate_fraction_min"]),
    }
    passed = all(gate_results.values())

    output.mkdir(parents=True)
    checkpoint = {
        "format": "strict-track2-action-source-gate-v1",
        "training_variant": "v15.7-hybrid-reset-window",
        "history_actions": HISTORY,
        "future_actions": FUTURE,
        "action_dim": ACTION_DIM,
        "feature_mean": torch.from_numpy(mean),
        "feature_std": torch.from_numpy(std),
        "hidden_features": int(config["hidden_features"]),
        "state_dict": model.state_dict(),
        "threshold": 0.5,
    }
    torch.save(checkpoint, output / "action_source_gate.pt")
    report = {
        "format": "strict-track2-v157-hybrid-action-gate-training-v1",
        "passed": passed,
        "gate_results": gate_results,
        "windows": str(root),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256(source_manifest),
        "preregistration": str(preregistration),
        "preregistration_sha256": sha256(preregistration),
        "probe_audits": [
            {"path": str(path), "sha256": sha256(path)} for path in probe_paths
        ],
        "feature_count": int(train_x.shape[1]),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "final_training_loss": loss_value,
        "train_arm_counts": train_arm_counts,
        "validation_arm_counts": validation_arm_counts,
        "train": train_metrics,
        "validation": validation_metrics,
        "probe_candidate_fraction": probe_candidate_fraction,
        "probes": probes,
        "actions_forwarded_unchanged": True,
        "participant_action_selection": False,
        "mpc": False,
    }
    (output / "training_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit("V15.7 hybrid action gate failed preregistered admission gates")


if __name__ == "__main__":
    main()
