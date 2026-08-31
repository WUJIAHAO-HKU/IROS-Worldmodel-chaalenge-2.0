#!/usr/bin/env python3
"""Train a public-only action-domain terminal rescue classifier."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from wam_pipeline.v396_action_phase_gate import FEATURE_VERSION, action_phase_features


SEED = 1556
CS = (0.001, 0.01, 0.1, 1.0)
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999)


def causal_action_features(history: np.ndarray, future: np.ndarray) -> np.ndarray:
    anchor = history[-1, 7:13]
    relative = future[:, 7:13] - anchor
    sequence = np.concatenate((anchor[None], future[:, 7:13]), axis=0)
    delta = np.diff(sequence, axis=0)
    path = np.linalg.norm(delta, axis=1)
    net = float(np.linalg.norm(relative[-1]))
    summary = np.asarray(
        [path.sum(), net, net / max(float(path.sum()), 1e-8), path.mean(), path.max(),
         np.linalg.norm(np.diff(delta, axis=0), axis=1).mean()], dtype=np.float32
    )
    gripper = np.concatenate((history[-1:, 13], future[:, 13])).astype(np.float32)
    return np.concatenate((relative.reshape(-1), delta.reshape(-1), gripper, summary))


class FrozenActionRoute:
    def __init__(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as values:
            self.mean = values["feature_mean"].astype(np.float32)
            self.scale = values["feature_scale"].astype(np.float32)
            self.coefficient = values["coefficient"].astype(np.float32)
            self.intercept = float(values["intercept"].item())

    def probability(self, history: np.ndarray, future: np.ndarray) -> float:
        feature = causal_action_features(history, future)
        logit = float(((feature - self.mean) / self.scale) @ self.coefficient + self.intercept)
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0))))

    def eligible(self, history: np.ndarray, future: np.ndarray) -> bool:
        probability = self.probability(history, future)
        post_grasp = bool(history[-1, 13] < 0.5 and (future[:, 13] < 0.5).mean() >= 0.75)
        sequence = np.concatenate((history[-1:, 7:13], future[:, 7:13]), axis=0)
        path = float(np.linalg.norm(np.diff(sequence, axis=0), axis=1).sum())
        failure = bool(
            (history[-1, 13] <= 0.5 and float(future[:, 13].mean()) > 0.5)
            or path <= 1e-6
            or (float(future[:, 13].mean()) <= 0.5 and probability < 0.01)
        )
        return bool(post_grasp and probability >= 0.99 and not failure)


def load_actions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return values["history_actions"].astype(np.float32), values["future_actions"].astype(np.float32)


def group_rates(y: np.ndarray, probability: np.ndarray, kind: np.ndarray, route: np.ndarray, threshold: float) -> dict:
    predicted = probability >= threshold

    def metrics(mask: np.ndarray) -> dict:
        positive = mask & (y == 1)
        negative = mask & (y == 0)
        return {
            "rows": int(mask.sum()), "positive": int(positive.sum()), "negative": int(negative.sum()),
            "positive_recall": float(predicted[positive].mean()) if positive.any() else None,
            "negative_specificity": float((~predicted[negative]).mean()) if negative.any() else None,
        }

    return {
        "threshold": threshold,
        "success_all": metrics(kind == "success_demo"),
        "success_route": metrics((kind == "success_demo") & route),
        "failure_all": metrics(kind == "failure_train"),
        "failure_route": metrics((kind == "failure_train") & route),
    }


def eligible(row: dict) -> bool:
    return bool(
        row["success_all"]["negative_specificity"] >= 0.98
        and row["success_route"]["negative_specificity"] >= 0.99
        and row["success_route"]["positive_recall"] >= 0.50
        and row["failure_all"]["negative_specificity"] >= 0.99
        and row["failure_route"]["negative_specificity"] >= 0.96
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("windows", "split", "onpolicy-windows", "onpolicy-split", "phase-gate", "action-gate", "preregistration", "output", "report"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing overwrite")
    prereg = json.loads(args.preregistration.read_text())
    if prereg.get("format") != "strict-track2-v396-action-domain-terminal-rescue-preregistration-v1":
        raise RuntimeError("wrong v396 preregistration")

    split = json.loads(args.split.read_text())
    train = set(map(int, split["train_episodes"]))
    arm = {int(key): value for key, value in split["arm_by_episode"].items()}
    success_episodes = sorted(episode for episode in train if arm[episode] == "right")
    with np.load(args.phase_gate, allow_pickle=False) as values:
        onset = {int(e) - 20000: int(s) for e, s in zip(values["episode"], values["onset_start"], strict=True)}
    if len(success_episodes) != 15 or set(onset) != set(success_episodes):
        raise RuntimeError("public success episode/onset drift")

    failure_split = json.loads(args.onpolicy_split.read_text())
    if failure_split.get("policy_source") != "official unmodified Pi0.5 baseline":
        raise RuntimeError("unexpected public failure policy source")
    failure_train = [r for r in failure_split["episodes"] if r["split"] == "train" and r["arm"] == "right"]
    failure_validation = [r for r in failure_split["episodes"] if r["split"] == "validation" and r["arm"] == "right"]
    if len(failure_train) != 47 or len(failure_validation) != 7:
        raise RuntimeError("public failure split drift")
    action_route = FrozenActionRoute(args.action_gate)

    features: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[int] = []
    kinds: list[str] = []
    routes: list[bool] = []

    def append(path: Path, label: int, group: int, kind: str) -> None:
        history, future = load_actions(path)
        features.append(action_phase_features(history, future))
        labels.append(label)
        groups.append(group)
        kinds.append(kind)
        routes.append(action_route.eligible(history, future))

    for episode in success_episodes:
        for path in sorted(args.windows.glob(f"episode{episode}_*.npz")):
            start = int(path.stem.split("_")[1])
            append(path, int(start >= onset[episode]), episode, "success_demo")
    success_rows = len(features)
    for row in failure_train:
        episode = int(row["episode_id"])
        paths = sorted(args.onpolicy_windows.glob(f"episode{episode}_*.npz"))
        if row["capture_success"] is not False or len(paths) != int(row["windows"]):
            raise RuntimeError(f"failure episode drift: {episode}")
        for path in paths:
            append(path, 0, episode, "failure_train")
    train_failure_rows = len(features) - success_rows

    validation_features: list[np.ndarray] = []
    validation_routes: list[bool] = []
    for row in failure_validation:
        episode = int(row["episode_id"])
        paths = sorted(args.onpolicy_windows.glob(f"episode{episode}_*.npz"))
        if row["capture_success"] is not False or len(paths) != int(row["windows"]):
            raise RuntimeError(f"validation failure episode drift: {episode}")
        for path in paths:
            history, future = load_actions(path)
            validation_features.append(action_phase_features(history, future))
            validation_routes.append(action_route.eligible(history, future))

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    group = np.asarray(groups, dtype=np.int64)
    kind = np.asarray(kinds)
    route = np.asarray(routes, dtype=bool)
    cv_rows = []
    for c_value in CS:
        oof = np.full(len(y), np.nan, dtype=np.float64)
        for train_idx, test_idx in GroupKFold(n_splits=15).split(x, y, groups=group):
            mean = x[train_idx].mean(axis=0)
            scale = np.maximum(x[train_idx].std(axis=0), 1e-6)
            model = LogisticRegression(C=c_value, max_iter=2000, class_weight="balanced", solver="liblinear", random_state=SEED)
            model.fit((x[train_idx] - mean) / scale, y[train_idx])
            oof[test_idx] = model.predict_proba((x[test_idx] - mean) / scale)[:, 1]
        for threshold in THRESHOLDS:
            row = group_rates(y, oof, kind, route, threshold)
            row["c"] = c_value
            row["eligible"] = eligible(row)
            cv_rows.append(row)

    candidates = [row for row in cv_rows if row["eligible"]]
    selected = max(candidates, key=lambda row: (row["success_route"]["positive_recall"], row["failure_route"]["negative_specificity"], row["success_route"]["negative_specificity"], row["threshold"], -row["c"])) if candidates else None
    checks = {
        "success_episodes_exact15": len(success_episodes) == 15,
        "success_rows_exact1926": success_rows == 1926,
        "failure_train_episodes_exact47": len(failure_train) == 47,
        "failure_train_rows_exact2256": train_failure_rows == 2256,
        "failure_validation_episodes_exact7": len(failure_validation) == 7,
        "failure_validation_rows_exact336": len(validation_features) == 336,
        "grouped_oof_complete": True,
        "eligible_operating_point": selected is not None,
    }
    artifact_sha256 = None
    final_rates = None
    validation_rates = None
    if selected is not None:
        mean = x.mean(axis=0)
        scale = np.maximum(x.std(axis=0), 1e-6)
        model = LogisticRegression(C=selected["c"], max_iter=2000, class_weight="balanced", solver="liblinear", random_state=SEED)
        model.fit((x - mean) / scale, y)
        probability = model.predict_proba((x - mean) / scale)[:, 1]
        final_rates = group_rates(y, probability, kind, route, selected["threshold"])
        vx = np.asarray(validation_features, dtype=np.float64)
        vr = np.asarray(validation_routes, dtype=bool)
        vp = model.predict_proba((vx - mean) / scale)[:, 1]
        predicted = vp >= selected["threshold"]
        validation_rates = {
            "rows": int(len(vx)), "route_rows": int(vr.sum()),
            "negative_specificity_all": float((~predicted).mean()),
            "negative_specificity_route": float((~predicted[vr]).mean()) if vr.any() else None,
            "maximum_probability": float(vp.max()),
            "route_maximum_probability": float(vp[vr].max()) if vr.any() else None,
        }
        checks["validation_route_rows_exact3"] = int(vr.sum()) == 3
        checks["validation_all_specificity_ge0p99"] = validation_rates["negative_specificity_all"] >= 0.99
        checks["validation_route_zero_false_positives"] = bool(vr.any() and validation_rates["negative_specificity_route"] == 1.0)
        if all(checks.values()):
            np.savez_compressed(args.output, feature_mean=mean.astype(np.float32), feature_scale=scale.astype(np.float32), coefficient=model.coef_[0].astype(np.float32), intercept=np.asarray(model.intercept_[0], dtype=np.float32), threshold=np.asarray(selected["threshold"], dtype=np.float32), c=np.asarray(selected["c"], dtype=np.float32), feature_version=np.asarray(FEATURE_VERSION))
            artifact_sha256 = hashlib.sha256(args.output.read_bytes()).hexdigest()

    report = {
        "format": "strict-track2-v396-action-domain-terminal-rescue-training-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "examples": int(len(y)), "feature_dimension": int(x.shape[1]), "positives": int(y.sum()),
        "success_rows": success_rows, "failure_train_rows": train_failure_rows,
        "route_rows": int(route.sum()), "selection": selected, "final_rates": final_rates,
        "validation_failure_rates": validation_rates, "cv_rows": cv_rows,
        "checks": checks, "passed": bool(all(checks.values()) and artifact_sha256),
        "artifact_sha256": artifact_sha256,
        "guards": {"public_data_only": True, "features_are_actions_only": True, "runtime_reads_reward_or_outcomes": False, "hidden_or_final_data": False, "real_submission": False},
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("examples", "feature_dimension", "selection", "validation_failure_rates", "checks", "passed", "artifact_sha256")}, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
