#!/usr/bin/env python3
"""Fit a monotone five-frame residual profile from public training pixels only."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from wam_pipeline.v342_temporal_blended_public_reanchor_runtime import (
    Track2V342TemporalBlendedPublicReanchor,
)
from wam_pipeline.v346_learned_temporal_residual_runtime import PROFILE_FORMAT


LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
FOLDS = 3
RGB_WEIGHT = 0.30
TEMPORAL_WEIGHT = 0.70


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_for(path: Path) -> int:
    return int.from_bytes(hashlib.sha256(path.name.encode()).digest()[:8], "little") % (2**31)


def collect(args, episodes: list[int], instructions: dict[str, str]):
    runtime = Track2V342TemporalBlendedPublicReanchor(
        args.checkpoint_dir, args.library_index, args.device,
        args.action_gate, args.phase_gate, args.recursive_ood_gate,
        reanchor_alpha=0.0, feature_workers=args.feature_workers,
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
                "instruction": str(instructions[str(episode)]),
            })
    samples = []
    rows_seen = 0
    while True:
        active = [state for state in states if state["start"] in state["available"]]
        if not active:
            break
        requests, contexts, histories, futures, seeds, prompts = [], [], [], [], [], []
        for state in active:
            path = state["available"][state["start"]]
            with np.load(path, allow_pickle=False) as payload:
                history = payload["history_actions"].astype(np.float32)
                future = payload["future_actions"].astype(np.float32)
                target = payload["target_frames"].astype(np.uint8)
            requests.append((state, path, future, target))
            contexts.append(state["recursive_context"])
            histories.append(history); futures.append(future)
            seeds.append(seed_for(path)); prompts.append(state["instruction"])
        predictions, masks, selected_rows = [], [], []
        for begin in range(0, len(contexts), args.inference_batch_size):
            end = begin + args.inference_batch_size
            batch = runtime.predict_batch(
                np.stack(contexts[begin:end]), np.stack(histories[begin:end]),
                np.stack(futures[begin:end]), np.asarray(seeds[begin:end], dtype=np.int64),
                prompts[begin:end],
            )
            predictions.extend(batch)
            masks.extend(runtime.last_recursive_reanchor_mask.tolist())
            selected_rows.extend(runtime.last_recursive_reanchor_rows.tolist())
        for index, (state, path, future, target) in enumerate(requests):
            baseline = predictions[index]
            if masks[index]:
                retrieval = runtime._target(int(selected_rows[index]))
                samples.append({
                    "episode": int(state["episode"]),
                    "key": f"episode{state['episode']}/a{state['alignment']}/s{state['start']:05d}",
                    "baseline": baseline[-5:].copy(),
                    "retrieval": retrieval[-5:].copy(),
                    "target": target[-5:].copy(),
                    "closed": (future[3:, 13] < 0.5).astype(np.float32),
                })
            state["recursive_context"] = baseline[-5:].copy()
            state["start"] += 8
            rows_seen += 1
    print(f"V346_PUBLIC_TRAIN_COLLECTED rows={rows_seen} direct={len(samples)}", flush=True)
    return samples, rows_seen


def quadratic_parts(samples: list[dict]):
    rgb_a = np.zeros((5, 5), dtype=np.float64)
    rgb_b = np.zeros(5, dtype=np.float64)
    temporal_a = np.zeros((5, 5), dtype=np.float64)
    temporal_b = np.zeros(5, dtype=np.float64)
    for sample in samples:
        base = sample["baseline"].astype(np.float64)
        delta = (sample["retrieval"].astype(np.float64) - base) * sample["closed"].reshape(5, 1, 1, 1)
        wanted = sample["target"].astype(np.float64) - base
        for frame in range(5):
            d = delta[frame].reshape(-1)
            y = wanted[frame].reshape(-1)
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


def fit(parts, temporal_weight: float) -> np.ndarray:
    rgb_a, rgb_b, temporal_a, temporal_b = parts
    a = rgb_a + temporal_weight * temporal_a
    b = rgb_b + temporal_weight * temporal_b
    scale = max(float(np.trace(a)) / 5.0, 1.0)
    a = a / scale + np.eye(5) * 1e-8
    b = b / scale
    constraints = [
        {"type": "ineq", "fun": lambda x, i=i: x[i + 1] - x[i]}
        for i in range(4)
    ]
    constraints.append({"type": "ineq", "fun": lambda x: x[4] - 0.75})
    result = minimize(
        lambda x: float(x @ a @ x - 2.0 * b @ x),
        np.linspace(0.45, 0.85, 5),
        jac=lambda x: 2.0 * (a @ x - b),
        method="SLSQP", bounds=[(0.0, 1.0)] * 5, constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"v346 constrained fit failed: {result.message}")
    return np.clip(result.x, 0.0, 1.0)


def metrics(samples: list[dict], coefficients: np.ndarray) -> dict:
    rgb, temporal, base_rgb, base_temporal = [], [], [], []
    shaped = coefficients.reshape(5, 1, 1, 1)
    for sample in samples:
        base = sample["baseline"].astype(np.float32)
        target = sample["target"].astype(np.float32)
        delta = (sample["retrieval"].astype(np.float32) - base) * sample["closed"].reshape(5, 1, 1, 1)
        output = np.clip(np.rint(base + shaped * delta), 0, 255)
        rgb.append(float(np.abs(output - target).mean()))
        temporal.append(float(np.abs(np.diff(output, axis=0) - np.diff(target, axis=0)).mean()))
        base_rgb.append(float(np.abs(base - target).mean()))
        base_temporal.append(float(np.abs(np.diff(base, axis=0) - np.diff(target, axis=0)).mean()))
    result = {
        "count": len(samples),
        "rgb_mae": float(np.mean(rgb)), "baseline_rgb_mae": float(np.mean(base_rgb)),
        "temporal_error": float(np.mean(temporal)),
        "baseline_temporal_error": float(np.mean(base_temporal)),
    }
    result["rgb_ratio"] = result["rgb_mae"] / max(result["baseline_rgb_mae"], 1e-9)
    result["temporal_ratio"] = result["temporal_error"] / max(result["baseline_temporal_error"], 1e-9)
    result["selection_score"] = RGB_WEIGHT * result["rgb_ratio"] + TEMPORAL_WEIGHT * result["temporal_ratio"]
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
        raise FileExistsError("v346 output already exists")
    prereg = json.loads(args.preregistration.read_text())
    if prereg.get("format") != "strict-track2-v347-public-temporal-residual-fit-preregistration-v1":
        raise RuntimeError("wrong v347 preregistration")
    mapping = json.loads(args.instruction_map.read_text())
    episodes = sorted(
        int(value) for value in mapping["train_episodes"]
        if mapping["arm_by_episode"][str(value)] == "right"
    )
    samples, rows_seen = collect(args, episodes, mapping["episode_to_instruction"])
    episode_fold = {episode: index % FOLDS for index, episode in enumerate(episodes)}
    episode_parts = {
        episode: quadratic_parts([s for s in samples if s["episode"] == episode])
        for episode in episodes
    }

    def combined_parts(selected_episodes):
        values = [np.zeros((5, 5)), np.zeros(5), np.zeros((5, 5)), np.zeros(5)]
        for episode in selected_episodes:
            for index, part in enumerate(episode_parts[episode]):
                values[index] += part
        return tuple(values)

    trials = []
    for temporal_weight in LAMBDAS:
        fold_results = []
        for fold in range(FOLDS):
            train = [s for s in samples if episode_fold[s["episode"]] != fold]
            held = [s for s in samples if episode_fold[s["episode"]] == fold]
            coefficients = fit(
                combined_parts([e for e in episodes if episode_fold[e] != fold]),
                temporal_weight,
            )
            fold_results.append({
                "fold": fold, "coefficients": coefficients.tolist(),
                "episodes": [e for e in episodes if episode_fold[e] == fold],
                **metrics(held, coefficients),
            })
        score = float(np.average(
            [value["selection_score"] for value in fold_results],
            weights=[value["count"] for value in fold_results],
        ))
        trials.append({"temporal_weight": temporal_weight, "oof_score": score, "folds": fold_results})
    selected = min(trials, key=lambda value: (value["oof_score"], value["temporal_weight"]))
    coefficients = fit(combined_parts(episodes), selected["temporal_weight"])
    all_metrics = metrics(samples, coefficients)
    selected_oof_rgb = float(np.average(
        [value["rgb_ratio"] for value in selected["folds"]],
        weights=[value["count"] for value in selected["folds"]],
    ))
    selected_oof_temporal = float(np.average(
        [value["temporal_ratio"] for value in selected["folds"]],
        weights=[value["count"] for value in selected["folds"]],
    ))
    counts = {str(episode): sum(s["episode"] == episode for s in samples) for episode in episodes}
    checks = {
        "exact_public_train_rows": rows_seen == 1926,
        "direct_samples_ge_100": len(samples) >= 100,
        "each_fold_direct_samples_ge_25": all(value["count"] >= 25 for value in selected["folds"]),
        "coefficients_finite": bool(np.all(np.isfinite(coefficients))),
        "coefficients_in_unit_interval": bool(np.all((coefficients >= 0) & (coefficients <= 1))),
        "coefficients_monotone": bool(np.all(np.diff(coefficients) >= -1e-6)),
        "terminal_coefficient_ge_0p75": bool(coefficients[-1] >= 0.75),
        "oof_rgb_ratio_le_0p90": selected_oof_rgb <= 0.90,
        "oof_temporal_ratio_le_0p90": selected_oof_temporal <= 0.90,
    }
    passed = all(checks.values())
    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_profile, coefficients=coefficients.astype(np.float32),
        temporal_weight=np.asarray(selected["temporal_weight"], dtype=np.float32),
    )
    manifest = {
        "format": PROFILE_FORMAT, "created_at": datetime.now(timezone.utc).isoformat(),
        "coefficients": coefficients.tolist(), "temporal_weight": selected["temporal_weight"],
        "fit_passed": passed,
        "guards": {"public_train_only": True, "outcomes_or_rewards_used": False,
                   "hidden_or_final_data": False, "official_batch16_outcomes": False},
        "profile_sha256": sha256(args.output_profile),
    }
    args.output_profile.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = {
        "format": "strict-track2-v347-public-temporal-residual-fit-report-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "episodes": episodes, "episode_fold": episode_fold, "direct_counts": counts,
        "rows_seen": rows_seen, "direct_samples": len(samples),
        "objective": {"rgb_weight": RGB_WEIGHT, "temporal_weight": TEMPORAL_WEIGHT},
        "lambda_trials": trials, "selected_temporal_weight": selected["temporal_weight"],
        "coefficients": coefficients.tolist(), "all_train_metrics": all_metrics,
        "selected_oof_rgb_ratio": selected_oof_rgb,
        "selected_oof_temporal_ratio": selected_oof_temporal,
        "checks": checks, "passed": passed,
        "authorizes_recursive_train_gate_only": passed,
        "evidence_sha256": {"preregistration": sha256(args.preregistration)},
        "guards": {"public_train_only": True, "outcomes_or_rewards_used": False,
                   "hidden_or_final_data": False, "real_submission": False},
    }
    args.output_report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "selected_temporal_weight": selected["temporal_weight"],
        "coefficients": coefficients.tolist(), "oof_rgb_ratio": selected_oof_rgb,
        "oof_temporal_ratio": selected_oof_temporal, "checks": checks, "passed": passed,
    }, indent=2), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
