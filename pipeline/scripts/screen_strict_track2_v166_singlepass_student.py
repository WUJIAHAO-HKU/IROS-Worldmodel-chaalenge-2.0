#!/usr/bin/env python3
"""Apply immutable offline admission gates to the V16.6 single-pass student."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def corr(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--visual", type=Path, required=True)
    parser.add_argument("--reward", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prereg = json.loads(args.preregistration.read_text())
    if prereg.get("format") != "strict-track2-v166-singlepass-student-screen-preregistration-v1":
        raise SystemExit("unexpected V16.6 screen preregistration")
    training = json.loads(args.training_manifest.read_text())
    visual = json.loads(args.visual.read_text())
    reward = json.loads(args.reward.read_text())
    if training.get("format") != "strict-track2-v166-singlepass-visual-student-training-v1":
        raise SystemExit("unexpected V16.6 training manifest")
    if visual.get("format") != "strict-track2-v166-singlepass-student-against-v157-cache-v1":
        raise SystemExit("unexpected V16.6 visual report")
    if reward.get("format") != "strict-track2-frozen-official-reward-alignment-audit-v1":
        raise SystemExit("unexpected official reward report")

    registered_cache = prereg["visual_screen"]
    if visual["v157_cache_sha256"] != registered_cache["v157_cache_sha256"]:
        raise SystemExit("visual report used a non-preregistered V15.7 cache")
    if int(visual["samples"]) != int(registered_cache["samples"]):
        raise SystemExit("visual report sample count differs from preregistration")
    candidate = args.candidate.resolve()
    if Path(visual["student"]).resolve() != candidate:
        raise SystemExit("visual report does not evaluate the selected candidate")
    if Path(reward["cache"]).resolve() != args.visual.with_name("v166_prediction_cache64.npz").resolve():
        raise SystemExit("reward report does not use the V16.6 prediction cache")

    initial = float(training["initial_ground_truth_validation_mae"])
    best = float(training["best_ground_truth_validation_mae"])
    validation_gain = 100.0 * (initial - best) / max(abs(initial), 1e-12)
    history_finite = all(
        math.isfinite(float(row["ground_truth_validation_mae"]))
        for row in training["validation"]
    )
    counts = training["source_counts"]
    visual_gain = visual["improvement"]
    reward_groups = reward["groups"]
    raw = {name: np.asarray(value, dtype=np.float64) for name, value in reward["raw_scores"].items()}
    with np.load(reward["reuse_baseline_cache"] or reward["cache"], allow_pickle=False) as values:
        right = values["arm_right"].astype(bool)
    terminal = {
        "overall_baseline": corr(raw["baseline"][:, -1], raw["target"][:, -1]),
        "overall_candidate": corr(raw["candidate"][:, -1], raw["target"][:, -1]),
        "right_baseline": corr(raw["baseline"][right, -1], raw["target"][right, -1]),
        "right_candidate": corr(raw["candidate"][right, -1], raw["target"][right, -1]),
    }
    latency = visual["latency_seconds"]
    memory = visual.get("cuda_memory") or {}
    batch_equivalence = visual.get("native_batch_equivalence") or {}
    gate = prereg["required_gates"]
    checks = {
        "pilot_validation_mae_ge_3": validation_gain >= gate["pilot_validation_mae_improvement_percent_min"],
        "validation_history_finite": history_finite,
        "validation_left_right_exact_16": training.get("validation_arm_counts") == {"left": 16, "right": 16},
        "effective_samples_exact_800": int(counts["teacher"]) + int(counts["ground_truth"]) == 800,
        "left_right_exactly_balanced": int(counts["left"]) == 400 and int(counts["right"]) == 400,
        "visual_overall_rgb": visual_gain["overall"]["rgb_mae_improvement_percent"] >= gate["visual_overall_rgb_improvement_percent_min"],
        "visual_left_rgb": visual_gain["left"]["rgb_mae_improvement_percent"] >= gate["visual_left_rgb_improvement_percent_min"],
        "visual_right_contact": visual_gain["right"]["contact_rgb_mae_improvement_percent"] >= gate["visual_right_contact_improvement_percent_min"],
        "visual_texture": visual_gain["overall"]["texture_mae_improvement_percent"] >= gate["visual_overall_texture_improvement_percent_min"],
        "visual_temporal": visual_gain["overall"]["temporal_delta_mae_improvement_percent"] >= gate["visual_overall_temporal_improvement_percent_min"],
        "reward_overall_mae": reward_groups["overall"]["improvement"]["reward_mae_improvement_percent"] >= gate["reward_overall_mae_improvement_percent_min"],
        "reward_right_mae": reward_groups["right"]["improvement"]["reward_mae_improvement_percent"] >= gate["reward_right_mae_improvement_percent_min"],
        "reward_overall_terminal_mae": reward_groups["overall"]["improvement"]["terminal_gain_mae_improvement_percent"] >= gate["reward_overall_terminal_gain_mae_improvement_percent_min"],
        "reward_right_terminal_mae": reward_groups["right"]["improvement"]["terminal_gain_mae_improvement_percent"] >= gate["reward_right_terminal_gain_mae_improvement_percent_min"],
        "reward_overall_terminal_pearson": terminal["overall_candidate"] >= terminal["overall_baseline"],
        "reward_right_terminal_pearson": terminal["right_candidate"] >= terminal["right_baseline"],
        "single_sample_latency": float(latency["steady_mean"]) <= gate["single_sample_steady_latency_seconds_max"],
        "batch32_sharded_latency": float(latency["batch32_sharded_max8"]) <= gate["batch32_sharded_max8_latency_seconds_max"],
        "peak_cuda_reserved": float(memory.get("peak_reserved_mib", math.inf)) <= gate["peak_cuda_reserved_mib_max"],
        "native_batch_uint8_mae": float(batch_equivalence.get("uint8_mae", math.inf)) <= gate["native_batch_uint8_mae_max"],
        "native_batch_max_error": int(batch_equivalence.get("uint8_max_absolute_error", 256)) <= gate["native_batch_uint8_max_absolute_error_max"],
    }
    report = {
        "format": "strict-track2-v166-singlepass-student-offline-screen-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "candidate": str(candidate),
        "pilot_validation_mae_improvement_percent": validation_gain,
        "terminal_pearson": terminal,
        "latency_seconds": latency,
        "cuda_memory": memory,
        "native_batch_equivalence": batch_equivalence,
        "evidence": {
            "preregistration": {"path": str(args.preregistration.resolve()), "sha256": sha256(args.preregistration)},
            "training_manifest": {"path": str(args.training_manifest.resolve()), "sha256": sha256(args.training_manifest)},
            "visual": {"path": str(args.visual.resolve()), "sha256": sha256(args.visual)},
            "reward": {"path": str(args.reward.resolve()), "sha256": sha256(args.reward)},
        },
        "interpretation_guard": prereg["promotion_guard"],
    }
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
