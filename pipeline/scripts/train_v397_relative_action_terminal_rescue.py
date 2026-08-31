#!/usr/bin/env python3
"""Train the relative-action-only v397 terminal rescue."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from train_v396_action_domain_terminal_rescue import FrozenActionRoute, group_rates, eligible, load_actions, CS, THRESHOLDS
from wam_pipeline.v397_relative_action_phase_gate import FEATURE_VERSION, relative_action_features


SEED = 1557


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("windows", "split", "onpolicy-windows", "onpolicy-split", "phase-gate", "action-gate", "preregistration", "output", "report"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing overwrite")
    if json.loads(args.preregistration.read_text()).get("format") != "strict-track2-v397-relative-action-terminal-rescue-preregistration-v1":
        raise RuntimeError("wrong v397 preregistration")

    split = json.loads(args.split.read_text())
    train = set(map(int, split["train_episodes"]))
    arm = {int(key): value for key, value in split["arm_by_episode"].items()}
    success_episodes = sorted(e for e in train if arm[e] == "right")
    with np.load(args.phase_gate, allow_pickle=False) as p:
        onset = {int(e) - 20000: int(s) for e, s in zip(p["episode"], p["onset_start"], strict=True)}
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
    features, labels, groups, kinds, routes = [], [], [], [], []

    def append(path: Path, label: int, group: int, kind: str) -> None:
        history, future = load_actions(path)
        features.append(relative_action_features(history, future))
        labels.append(label); groups.append(group); kinds.append(kind)
        routes.append(action_route.eligible(history, future))

    for episode in success_episodes:
        for path in sorted(args.windows.glob(f"episode{episode}_*.npz")):
            append(path, int(int(path.stem.split("_")[1]) >= onset[episode]), episode, "success_demo")
    success_rows = len(features)
    for row in failure_train:
        episode = int(row["episode_id"])
        paths = sorted(args.onpolicy_windows.glob(f"episode{episode}_*.npz"))
        if row["capture_success"] is not False or len(paths) != int(row["windows"]):
            raise RuntimeError(f"failure episode drift: {episode}")
        for path in paths:
            append(path, 0, episode, "failure_train")
    failure_rows = len(features) - success_rows

    validation_features, validation_routes = [], []
    for row in failure_validation:
        episode = int(row["episode_id"])
        paths = sorted(args.onpolicy_windows.glob(f"episode{episode}_*.npz"))
        if row["capture_success"] is not False or len(paths) != int(row["windows"]):
            raise RuntimeError(f"validation failure episode drift: {episode}")
        for path in paths:
            history, future = load_actions(path)
            validation_features.append(relative_action_features(history, future))
            validation_routes.append(action_route.eligible(history, future))

    x = np.asarray(features, np.float64); y = np.asarray(labels, np.int64)
    group = np.asarray(groups, np.int64); kind = np.asarray(kinds); route = np.asarray(routes, bool)
    cv_rows = []
    for c_value in CS:
        oof = np.full(len(y), np.nan)
        for train_idx, test_idx in GroupKFold(n_splits=15).split(x, y, groups=group):
            mean = x[train_idx].mean(0); scale = np.maximum(x[train_idx].std(0), 1e-6)
            model = LogisticRegression(C=c_value, max_iter=2000, class_weight="balanced", solver="liblinear", random_state=SEED)
            model.fit((x[train_idx] - mean) / scale, y[train_idx])
            oof[test_idx] = model.predict_proba((x[test_idx] - mean) / scale)[:, 1]
        for threshold in THRESHOLDS:
            row = group_rates(y, oof, kind, route, threshold)
            row["c"] = c_value; row["eligible"] = eligible(row); cv_rows.append(row)
    candidates = [r for r in cv_rows if r["eligible"]]
    selected = max(candidates, key=lambda r: (r["success_route"]["positive_recall"], r["failure_route"]["negative_specificity"], r["success_route"]["negative_specificity"], r["threshold"], -r["c"])) if candidates else None
    checks = {
        "success_episodes_exact15": len(success_episodes) == 15, "success_rows_exact1926": success_rows == 1926,
        "failure_train_episodes_exact47": len(failure_train) == 47, "failure_train_rows_exact2256": failure_rows == 2256,
        "failure_validation_episodes_exact7": len(failure_validation) == 7, "failure_validation_rows_exact336": len(validation_features) == 336,
        "grouped_oof_complete": True, "eligible_operating_point": selected is not None,
    }
    artifact_sha256 = None; final_rates = None; validation_rates = None
    if selected:
        mean = x.mean(0); scale = np.maximum(x.std(0), 1e-6)
        model = LogisticRegression(C=selected["c"], max_iter=2000, class_weight="balanced", solver="liblinear", random_state=SEED)
        model.fit((x - mean) / scale, y)
        final_rates = group_rates(y, model.predict_proba((x - mean) / scale)[:, 1], kind, route, selected["threshold"])
        vx = np.asarray(validation_features, np.float64); vr = np.asarray(validation_routes, bool)
        vp = model.predict_proba((vx - mean) / scale)[:, 1]; predicted = vp >= selected["threshold"]
        validation_rates = {"rows": int(len(vx)), "route_rows": int(vr.sum()), "negative_specificity_all": float((~predicted).mean()), "negative_specificity_route": float((~predicted[vr]).mean()) if vr.any() else None, "maximum_probability": float(vp.max()), "route_maximum_probability": float(vp[vr].max()) if vr.any() else None}
        checks.update({"validation_route_rows_exact3": int(vr.sum()) == 3, "validation_all_specificity_ge0p99": validation_rates["negative_specificity_all"] >= .99, "validation_route_zero_false_positives": bool(vr.any() and validation_rates["negative_specificity_route"] == 1.0)})
        if all(checks.values()):
            np.savez_compressed(args.output, feature_mean=mean.astype(np.float32), feature_scale=scale.astype(np.float32), coefficient=model.coef_[0].astype(np.float32), intercept=np.asarray(model.intercept_[0], np.float32), threshold=np.asarray(selected["threshold"], np.float32), c=np.asarray(selected["c"], np.float32), feature_version=np.asarray(FEATURE_VERSION))
            artifact_sha256 = hashlib.sha256(args.output.read_bytes()).hexdigest()
    report = {"format": "strict-track2-v397-relative-action-terminal-rescue-training-v1", "created_at": datetime.now(timezone.utc).isoformat(), "examples": int(len(y)), "feature_dimension": int(x.shape[1]), "positives": int(y.sum()), "success_rows": success_rows, "failure_train_rows": failure_rows, "route_rows": int(route.sum()), "selection": selected, "final_rates": final_rates, "validation_failure_rates": validation_rates, "cv_rows": cv_rows, "checks": checks, "passed": bool(all(checks.values()) and artifact_sha256), "artifact_sha256": artifact_sha256, "guards": {"public_data_only": True, "features_are_relative_actions_only": True, "absolute_pose_used": False, "rgb_used": False, "runtime_reads_reward_or_outcomes": False, "hidden_or_final_data": False, "real_submission": False}}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("examples", "feature_dimension", "selection", "validation_failure_rates", "checks", "passed", "artifact_sha256")}, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
