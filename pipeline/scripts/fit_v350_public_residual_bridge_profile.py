#!/usr/bin/env python3
"""Fit the v350 start-aligned residual bridge on public train pixels only."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import fit_v346_public_temporal_residual_profile as shared
from wam_pipeline.v350_learned_residual_bridge_runtime import PROFILE_FORMAT


def parts(samples):
    rgb_a = np.zeros((5, 5), dtype=np.float64)
    rgb_b = np.zeros(5, dtype=np.float64)
    temporal_a = np.zeros((5, 5), dtype=np.float64)
    temporal_b = np.zeros(5, dtype=np.float64)
    for sample in samples:
        baseline = sample["baseline"].astype(np.float64)
        retrieval = sample["retrieval"].astype(np.float64)
        target = sample["target"].astype(np.float64)
        closed = sample["closed"].reshape(5, 1, 1, 1)
        base = closed * retrieval + (1.0 - closed) * baseline
        delta = closed * (baseline[0] - retrieval[0])
        wanted = target - base
        for frame in range(5):
            d = delta[frame].reshape(-1); y = wanted[frame].reshape(-1)
            rgb_a[frame, frame] += np.dot(d, d)
            rgb_b[frame] += np.dot(d, y)
        for frame in range(1, 5):
            current = delta[frame].reshape(-1)
            previous = delta[frame - 1].reshape(-1)
            y = (wanted[frame] - wanted[frame - 1]).reshape(-1)
            temporal_a[frame, frame] += np.dot(current, current)
            temporal_a[frame - 1, frame - 1] += np.dot(previous, previous)
            cross = np.dot(current, previous)
            temporal_a[frame, frame - 1] -= cross
            temporal_a[frame - 1, frame] -= cross
            temporal_b[frame] += np.dot(current, y)
            temporal_b[frame - 1] -= np.dot(previous, y)
    return rgb_a, rgb_b, temporal_a, temporal_b


def combine(episode_parts, episodes):
    result = [np.zeros((5, 5)), np.zeros(5), np.zeros((5, 5)), np.zeros(5)]
    for episode in episodes:
        for index, value in enumerate(episode_parts[episode]):
            result[index] += value
    return tuple(result)


def fit(values, temporal_weight):
    rgb_a, rgb_b, temporal_a, temporal_b = values
    a = rgb_a + temporal_weight * temporal_a
    b = rgb_b + temporal_weight * temporal_b
    scale = max(float(np.trace(a)) / 5.0, 1.0)
    a = a / scale + np.eye(5) * 1e-8
    b = b / scale
    constraints = [
        {"type": "ineq", "fun": lambda x, i=i: x[i] - x[i + 1]}
        for i in range(4)
    ]
    constraints.extend((
        {"type": "ineq", "fun": lambda x: x[0] - 0.75},
        {"type": "ineq", "fun": lambda x: 0.25 - x[4]},
    ))
    result = minimize(
        lambda x: float(x @ a @ x - 2.0 * b @ x),
        np.linspace(0.9, 0.1, 5), jac=lambda x: 2.0 * (a @ x - b),
        method="SLSQP", bounds=[(0.0, 1.0)] * 5, constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"v350 constrained fit failed: {result.message}")
    return np.clip(result.x, 0.0, 1.0)


def metrics(samples, coefficients):
    rgb, temporal, base_rgb, base_temporal = [], [], [], []
    shaped = coefficients.reshape(5, 1, 1, 1)
    for sample in samples:
        baseline = sample["baseline"].astype(np.float32)
        retrieval = sample["retrieval"].astype(np.float32)
        target = sample["target"].astype(np.float32)
        closed = sample["closed"].reshape(5, 1, 1, 1)
        bridge = retrieval + shaped * (baseline[0] - retrieval[0])
        output = np.where(closed, np.clip(np.rint(bridge), 0, 255), baseline)
        rgb.append(float(np.abs(output - target).mean()))
        temporal.append(float(np.abs(np.diff(output, axis=0) - np.diff(target, axis=0)).mean()))
        base_rgb.append(float(np.abs(baseline - target).mean()))
        base_temporal.append(float(np.abs(np.diff(baseline, axis=0) - np.diff(target, axis=0)).mean()))
    result = {
        "count": len(samples), "rgb_mae": float(np.mean(rgb)),
        "temporal_error": float(np.mean(temporal)),
        "baseline_rgb_mae": float(np.mean(base_rgb)),
        "baseline_temporal_error": float(np.mean(base_temporal)),
    }
    result["rgb_ratio"] = result["rgb_mae"] / max(result["baseline_rgb_mae"], 1e-9)
    result["temporal_ratio"] = result["temporal_error"] / max(result["baseline_temporal_error"], 1e-9)
    result["selection_score"] = shared.RGB_WEIGHT * result["rgb_ratio"] + shared.TEMPORAL_WEIGHT * result["temporal_ratio"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "checkpoint-dir", "library-index", "action-gate", "phase-gate", "recursive-ood-gate",
        "windows", "instruction-map", "preregistration", "output-profile", "output-report",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--feature-workers", type=int, default=8)
    args = parser.parse_args()
    if args.output_profile.exists() or args.output_report.exists():
        raise FileExistsError("v350 output already exists")
    prereg = json.loads(args.preregistration.read_text())
    if prereg.get("format") != "strict-track2-v350-public-residual-bridge-fit-preregistration-v1":
        raise RuntimeError("wrong v350 preregistration")
    mapping = json.loads(args.instruction_map.read_text())
    episodes = sorted(int(e) for e in mapping["train_episodes"] if mapping["arm_by_episode"][str(e)] == "right")
    samples, rows_seen = shared.collect(args, episodes, mapping["episode_to_instruction"])
    episode_fold = {episode: index % shared.FOLDS for index, episode in enumerate(episodes)}
    episode_parts = {episode: parts([s for s in samples if s["episode"] == episode]) for episode in episodes}
    trials = []
    for temporal_weight in shared.LAMBDAS:
        folds = []
        for fold in range(shared.FOLDS):
            train_episodes = [e for e in episodes if episode_fold[e] != fold]
            held = [s for s in samples if episode_fold[s["episode"]] == fold]
            coefficients = fit(combine(episode_parts, train_episodes), temporal_weight)
            folds.append({"fold": fold, "coefficients": coefficients.tolist(),
                          "episodes": [e for e in episodes if episode_fold[e] == fold],
                          **metrics(held, coefficients)})
        score = float(np.average([v["selection_score"] for v in folds], weights=[v["count"] for v in folds]))
        trials.append({"temporal_weight": temporal_weight, "oof_score": score, "folds": folds})
    selected = min(trials, key=lambda value: (value["oof_score"], value["temporal_weight"]))
    coefficients = fit(combine(episode_parts, episodes), selected["temporal_weight"])
    oof_rgb = float(np.average([v["rgb_ratio"] for v in selected["folds"]], weights=[v["count"] for v in selected["folds"]]))
    oof_temporal = float(np.average([v["temporal_ratio"] for v in selected["folds"]], weights=[v["count"] for v in selected["folds"]]))
    checks = {
        "exact_public_train_rows": rows_seen == 1926, "direct_samples_ge_100": len(samples) >= 100,
        "each_fold_direct_samples_ge_25": all(v["count"] >= 25 for v in selected["folds"]),
        "coefficients_finite": bool(np.all(np.isfinite(coefficients))),
        "coefficients_in_unit_interval": bool(np.all((coefficients >= 0) & (coefficients <= 1))),
        "coefficients_nonincreasing": bool(np.all(np.diff(coefficients) <= 1e-6)),
        "start_coefficient_ge_0p75": bool(coefficients[0] >= 0.75),
        "terminal_coefficient_le_0p25": bool(coefficients[-1] <= 0.25),
        "oof_rgb_ratio_le_0p90": oof_rgb <= 0.90,
        "oof_temporal_ratio_le_0p90": oof_temporal <= 0.90,
    }
    passed = all(checks.values())
    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_profile, bridge_coefficients=coefficients.astype(np.float32),
                        temporal_weight=np.asarray(selected["temporal_weight"], dtype=np.float32))
    manifest = {
        "format": PROFILE_FORMAT, "created_at": datetime.now(timezone.utc).isoformat(),
        "bridge_coefficients": coefficients.tolist(), "temporal_weight": selected["temporal_weight"],
        "fit_passed": passed,
        "guards": {"public_train_only": True, "outcomes_or_rewards_used": False,
                   "hidden_or_final_data": False, "official_batch16_outcomes": False},
        "profile_sha256": shared.sha256(args.output_profile),
    }
    args.output_profile.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = {
        "format": "strict-track2-v350-public-residual-bridge-fit-report-v1",
        "created_at": datetime.now(timezone.utc).isoformat(), "rows_seen": rows_seen,
        "direct_samples": len(samples), "episode_fold": episode_fold,
        "lambda_trials": trials, "selected_temporal_weight": selected["temporal_weight"],
        "bridge_coefficients": coefficients.tolist(), "all_train_metrics": metrics(samples, coefficients),
        "selected_oof_rgb_ratio": oof_rgb, "selected_oof_temporal_ratio": oof_temporal,
        "checks": checks, "passed": passed, "authorizes_recursive_train_gate_only": passed,
        "guards": {"public_train_only": True, "outcomes_or_rewards_used": False,
                   "hidden_or_final_data": False, "real_submission": False},
    }
    args.output_report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected_temporal_weight": selected["temporal_weight"],
                      "bridge_coefficients": coefficients.tolist(), "oof_rgb_ratio": oof_rgb,
                      "oof_temporal_ratio": oof_temporal, "checks": checks, "passed": passed}, indent=2), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
