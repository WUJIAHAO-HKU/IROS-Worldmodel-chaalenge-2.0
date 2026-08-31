#!/usr/bin/env python3
"""Search a routed V16.6 blend while preserving V15.7 terminal frames.

This is an offline world-model admission screen.  It never selects or edits a
policy action.  The selected profile is fixed before any real RoboTwin policy
evaluation and therefore cannot act as MPC.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_strict_track2_reward_alignment import (
    correlation,
    metrics,
    score_sequences,
    select_prompts,
)


def parse_alphas(values: list[float]) -> list[float]:
    result = sorted(set(float(value) for value in values))
    if not result or any(value < 0.0 or value > 1.0 for value in result):
        raise ValueError("blend alphas must be in [0, 1]")
    return result


def visual_metrics(predicted: np.ndarray, target: np.ndarray) -> dict:
    error = np.abs(predicted.astype(np.float32) - target.astype(np.float32))
    return {
        "rgb_mae": float(error.mean()),
        "frame_rgb_mae": error.mean(axis=(0, 2, 3, 4)).tolist(),
    }


def visual_comparison(baseline: dict, candidate: dict) -> dict:
    denominator = max(abs(float(baseline["rgb_mae"])), 1e-12)
    frame_baseline = np.asarray(baseline["frame_rgb_mae"], dtype=np.float64)
    frame_candidate = np.asarray(candidate["frame_rgb_mae"], dtype=np.float64)
    return {
        "rgb_mae_improvement_percent": 100.0
        * (float(baseline["rgb_mae"]) - float(candidate["rgb_mae"]))
        / denominator,
        "frame_rgb_mae_improvement_percent": (
            100.0 * (frame_baseline - frame_candidate) / np.maximum(np.abs(frame_baseline), 1e-12)
        ).tolist(),
    }


def make_prediction(
    baseline: np.ndarray,
    student: np.ndarray,
    arm_right: np.ndarray,
    middle_alpha: float,
    terminal_alpha: float,
    shape: str,
) -> np.ndarray:
    alpha = np.zeros((len(baseline), baseline.shape[1], 1, 1, 1), dtype=np.float32)
    # Frames are zero-indexed: preserve t+1/t+2, blend t+3..t+6, and use a
    # separately searched (normally zero) terminal coefficient for t+7/t+8.
    profiles = {
        "plateau": np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        "tapered": np.asarray([0.5, 1.0, 1.0, 0.5], dtype=np.float32),
        "triangular": np.asarray([0.25, 0.75, 0.75, 0.25], dtype=np.float32),
        "cosine": np.asarray([0.1464466, 0.5, 0.5, 0.1464466], dtype=np.float32),
    }
    if shape not in profiles:
        raise ValueError(f"unknown blend shape: {shape}")
    alpha[arm_right, 2:6, 0, 0, 0] = middle_alpha * profiles[shape]
    alpha[arm_right, 6:8] = terminal_alpha
    mixed = baseline.astype(np.float32) + alpha * (
        student.astype(np.float32) - baseline.astype(np.float32)
    )
    return np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--reward-checkpoint", required=True, type=Path)
    parser.add_argument("--t5-model", required=True, type=Path)
    parser.add_argument("--reset-manifest", required=True, type=Path)
    parser.add_argument("--instruction-map", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--selected-cache", required=True, type=Path)
    parser.add_argument("--middle-alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--terminal-alphas", type=float, nargs="+", default=[0.0])
    parser.add_argument(
        "--shapes",
        nargs="+",
        choices=("plateau", "tapered", "triangular", "cosine"),
        default=["plateau", "tapered", "triangular", "cosine"],
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    middle_alphas = parse_alphas(args.middle_alphas)
    terminal_alphas = parse_alphas(args.terminal_alphas)
    if len(terminal_alphas) != 1:
        raise ValueError("this terminal-protection screen requires exactly one terminal alpha")
    if args.output.exists() or args.selected_cache.exists():
        raise SystemExit("refusing to overwrite V16.8 search output")

    with np.load(args.cache, allow_pickle=False) as values:
        arrays = {key: values[key].copy() for key in values.files}
    required = {
        "context_last", "target", "baseline", "candidate", "arm_right",
        "route_right", "synthetic_seed",
    }
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"cache is missing arrays: {sorted(missing)}")
    baseline = arrays["baseline"]
    student = arrays["candidate"]
    target = arrays["target"]
    context = arrays["context_last"]
    arm_right = arrays["arm_right"].astype(bool)
    route_right = arrays["route_right"].astype(bool)
    if baseline.shape != student.shape or baseline.shape != target.shape:
        raise ValueError("baseline, student and target tensors must have identical shapes")
    if baseline.ndim != 5 or baseline.shape[1] != 8 or baseline.dtype != np.uint8:
        raise ValueError("expected [window,8,height,width,3] uint8 predictions")
    if not arm_right.any() or arm_right.all() or not route_right.any() or route_right.all():
        raise ValueError("cache must contain both arms")

    instruction_doc = json.loads(args.instruction_map.read_text(encoding="utf-8"))
    raw_mapping = instruction_doc.get("seed_to_instruction", instruction_doc)
    instruction_mapping = {int(seed): str(value) for seed, value in raw_mapping.items()}
    synthetic_seed = arrays["synthetic_seed"].astype(np.int64)
    missing_seeds = sorted(set(synthetic_seed.tolist()).difference(instruction_mapping))
    if missing_seeds:
        raise ValueError(f"instruction map is missing seeds: {missing_seeds}")
    prompts = [[instruction_mapping[int(seed)]] for seed in synthetic_seed]

    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel

    device = torch.device(args.device)
    reward_model = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        args.reward_checkpoint,
        config={"t5_model_name": str(args.t5_model)},
    ).to(device).eval().requires_grad_(False)
    scores = {
        "context": score_sequences(reward_model, context[:, None], prompts, device, args.batch_size),
        "target": score_sequences(reward_model, target, prompts, device, args.batch_size),
        "baseline": score_sequences(reward_model, baseline, prompts, device, args.batch_size),
    }

    masks = {
        "overall": np.ones(len(arm_right), dtype=bool),
        "left": ~arm_right,
        "right": arm_right,
    }
    if "capture_success" in arrays:
        success = arrays["capture_success"].astype(bool)
        masks["capture_success"] = success
        masks["capture_failure"] = ~success
    baseline_visual = {
        name: visual_metrics(baseline[mask], target[mask])
        for name, mask in masks.items()
        if mask.any()
    }
    baseline_reward = {
        name: metrics(scores["baseline"][mask], scores["target"][mask], scores["context"][mask, 0])
        for name, mask in masks.items()
        if mask.any()
    }

    rows = []
    predictions: dict[str, np.ndarray] = {}
    for shape in args.shapes:
        for middle_alpha in middle_alphas:
            terminal_alpha = terminal_alphas[0]
            tag = f"{shape}_middle_{middle_alpha:.4f}_terminal_{terminal_alpha:.4f}"
            prediction = make_prediction(
                baseline, student, route_right, middle_alpha, terminal_alpha, shape
            )
            candidate_score = score_sequences(
                reward_model, prediction, prompts, device, args.batch_size
            )
            visual = {}
            reward = {}
            for name, mask in masks.items():
                if not mask.any():
                    continue
                candidate_visual = visual_metrics(prediction[mask], target[mask])
                candidate_reward = metrics(
                    candidate_score[mask], scores["target"][mask], scores["context"][mask, 0]
                )
                visual[name] = {
                    "baseline": baseline_visual[name],
                    "candidate": candidate_visual,
                    "improvement": visual_comparison(baseline_visual[name], candidate_visual),
                }
                reward[name] = {
                    "baseline": baseline_reward[name],
                    "candidate": candidate_reward,
                }
            terminal = {
                "right_baseline_pearson": correlation(
                    scores["baseline"][arm_right, -1], scores["target"][arm_right, -1]
                ),
                "right_candidate_pearson": correlation(
                    candidate_score[arm_right, -1], scores["target"][arm_right, -1]
                ),
            }
            tolerance = 1e-12
            checks = {
                "unrouted_bit_exact": bool(np.array_equal(prediction[~route_right], baseline[~route_right])),
                "routed_first_two_bit_exact": bool(
                    np.array_equal(prediction[route_right, :2], baseline[route_right, :2])
                ),
                "routed_terminal_bit_exact": bool(
                    np.array_equal(prediction[route_right, 6:], baseline[route_right, 6:])
                ),
                "overall_visual_nonregression": visual["overall"]["improvement"]["rgb_mae_improvement_percent"] >= 0.0,
                "left_visual_nonregression": visual["left"]["improvement"]["rgb_mae_improvement_percent"] >= 0.0,
                "right_visual_improves": visual["right"]["improvement"]["rgb_mae_improvement_percent"] > 0.0,
                "overall_reward_mae_nonregression": reward["overall"]["candidate"]["reward_mae_normalized_by_gt_std"] <= reward["overall"]["baseline"]["reward_mae_normalized_by_gt_std"] + tolerance,
                "left_reward_mae_nonregression": reward["left"]["candidate"]["reward_mae_normalized_by_gt_std"] <= reward["left"]["baseline"]["reward_mae_normalized_by_gt_std"] + tolerance,
                "right_reward_mae_improves": reward["right"]["candidate"]["reward_mae_normalized_by_gt_std"] < reward["right"]["baseline"]["reward_mae_normalized_by_gt_std"],
                "overall_reward_pearson_nonregression": reward["overall"]["candidate"]["flattened_pearson"] >= reward["overall"]["baseline"]["flattened_pearson"] - tolerance,
                "left_reward_pearson_nonregression": reward["left"]["candidate"]["flattened_pearson"] >= reward["left"]["baseline"]["flattened_pearson"] - tolerance,
                "right_reward_pearson_nonregression": reward["right"]["candidate"]["flattened_pearson"] >= reward["right"]["baseline"]["flattened_pearson"] - tolerance,
                "right_reward_delta_mae_nonregression": reward["right"]["candidate"]["delta_mae"] <= reward["right"]["baseline"]["delta_mae"] + tolerance,
                "right_terminal_gain_mae_nonregression": reward["right"]["candidate"]["terminal_gain_mae"] <= reward["right"]["baseline"]["terminal_gain_mae"] + tolerance,
                "right_terminal_sign_nonregression": reward["right"]["candidate"]["terminal_gain_sign_agreement"] >= reward["right"]["baseline"]["terminal_gain_sign_agreement"] - tolerance,
                "right_terminal_pearson_nonregression": terminal["right_candidate_pearson"] >= terminal["right_baseline_pearson"] - tolerance,
            }
            row = {
                "tag": tag,
                "profile": {
                    "left_alpha": 0.0,
                    "right_first_two_alpha": 0.0,
                    "right_middle_shape": shape,
                    "right_middle_t3_t6_alpha": middle_alpha,
                    "right_terminal_t7_t8_alpha": terminal_alpha,
                },
                "passed": all(checks.values()),
                "checks": checks,
                "failed_checks": [name for name, passed in checks.items() if not passed],
                "visual": visual,
                "reward": reward,
                "terminal": terminal,
            }
            rows.append(row)
            predictions[tag] = prediction
            print(json.dumps({"profile": row["profile"], "passed": row["passed"], "failed": row["failed_checks"]}), flush=True)

    eligible = [row for row in rows if row["passed"]]
    eligible.sort(
        key=lambda row: (
            row["reward"]["right"]["baseline"]["reward_mae_normalized_by_gt_std"]
            - row["reward"]["right"]["candidate"]["reward_mae_normalized_by_gt_std"],
            row["visual"]["right"]["improvement"]["rgb_mae_improvement_percent"],
            row["reward"]["overall"]["baseline"]["reward_mae_normalized_by_gt_std"]
            - row["reward"]["overall"]["candidate"]["reward_mae_normalized_by_gt_std"],
        ),
        reverse=True,
    )
    selected = eligible[0] if eligible else None
    report = {
        "format": "strict-track2-v169-routed-terminal-protected-blend-search-v1",
        "classification": "offline fixed world-model profile search; no policy action selection",
        "cache": str(args.cache.resolve()),
        "route_audit": {
            "accuracy": float((route_right == arm_right).mean()),
            "confusion": [
                [int(np.count_nonzero((arm_right == truth) & (route_right == guess))) for guess in (False, True)]
                for truth in (False, True)
            ],
        },
        "profiles": rows,
        "eligible_profiles": [row["tag"] for row in eligible],
        "selected_profile": selected["tag"] if selected else None,
        "promotion_guard": "A selected profile permits only an official one-update RL diagnostic. It is not Track 2 completion evidence.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if selected is None:
        raise SystemExit("no terminal-protected blend passed all gates")
    selected_prediction = predictions[selected["tag"]]
    cache_output = {key: value for key, value in arrays.items() if key != "candidate"}
    cache_output["candidate"] = selected_prediction
    args.selected_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.selected_cache, **cache_output)
    args.selected_cache.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "format": "strict-track2-v169-routed-terminal-protected-blend-cache-v1",
                "source_cache": str(args.cache.resolve()),
                "selected_profile": selected["profile"],
                "left_and_terminal_protection": {
                    "unrouted_bit_exact": selected["checks"]["unrouted_bit_exact"],
                    "routed_first_two_bit_exact": selected["checks"]["routed_first_two_bit_exact"],
                    "routed_terminal_bit_exact": selected["checks"]["routed_terminal_bit_exact"],
                },
                "interpretation_guard": report["promotion_guard"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"selected_profile": selected["profile"], "selected_cache": str(args.selected_cache.resolve())}, indent=2))


if __name__ == "__main__":
    main()
