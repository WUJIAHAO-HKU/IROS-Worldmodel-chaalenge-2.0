#!/usr/bin/env python3
"""Audit whether a frozen Track 2 reward model preserves GT progress on rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def select_prompts(manifest: Path, arm_right: bool, limit: int) -> list[str]:
    rows = json.loads(manifest.read_text())["episodes"]
    side = "right arm" if arm_right else "left arm"
    prompts = sorted({str(row["instruction"]) for row in rows if side in str(row["instruction"]).lower()})
    if not prompts:
        raise ValueError(f"no {side} prompts in {manifest}")
    if len(prompts) <= limit:
        return prompts
    indices = np.linspace(0, len(prompts) - 1, limit, dtype=int)
    return [prompts[index] for index in indices]


def score_sequences(model, frames: np.ndarray, prompts: list[list[str]], device, batch_size: int) -> np.ndarray:
    # frames: [window,time,height,width,channel]. Each image is scored under
    # several arm-consistent official prompts; their mean is the robust score.
    window_count, time = frames.shape[:2]
    output = np.empty((window_count, time), dtype=np.float64)
    jobs: list[tuple[int, int, str]] = []
    for window in range(window_count):
        for frame in range(time):
            for prompt in prompts[window]:
                jobs.append((window, frame, prompt))
    sums = np.zeros_like(output)
    counts = np.zeros_like(output)
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start : start + batch_size]
        images = np.stack([frames[w, t] for w, t, _ in batch])
        tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float().div(255).to(device)
        instructions = [prompt for _, _, prompt in batch]
        # Match Track2HttpEnv: the official reward checkpoint runs in FP32.
        with torch.inference_mode():
            scores = model.compute_reward(tensor, instructions).float().cpu().numpy()
        for (window, frame, _), value in zip(batch, scores):
            sums[window, frame] += float(value)
            counts[window, frame] += 1
    return sums / counts


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def metrics(predicted: np.ndarray, target: np.ndarray, context: np.ndarray) -> dict[str, float]:
    error = predicted - target
    target_delta = np.diff(np.concatenate((context[:, None], target), axis=1), axis=1)
    predicted_delta = np.diff(np.concatenate((context[:, None], predicted), axis=1), axis=1)
    target_gain = target[:, -1] - context
    predicted_gain = predicted[:, -1] - context
    target_scale = max(float(np.std(target)), 1e-12)
    per_window_corr = [correlation(p, t) for p, t in zip(predicted, target)]
    active = np.abs(target_delta) > max(float(np.std(target_delta)) * 0.1, 1e-9)
    return {
        "reward_mae": float(np.mean(np.abs(error))),
        "reward_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "reward_mae_normalized_by_gt_std": float(np.mean(np.abs(error)) / target_scale),
        "flattened_pearson": correlation(predicted.ravel(), target.ravel()),
        "mean_per_window_pearson": float(np.mean(per_window_corr)),
        "delta_mae": float(np.mean(np.abs(predicted_delta - target_delta))),
        "delta_sign_agreement": float(np.mean(np.sign(predicted_delta[active]) == np.sign(target_delta[active]))) if active.any() else 0.0,
        "terminal_gain_mae": float(np.mean(np.abs(predicted_gain - target_gain))),
        "terminal_gain_sign_agreement": float(np.mean(np.sign(predicted_gain) == np.sign(target_gain))),
        "dynamic_range_ratio": float(np.std(predicted) / target_scale),
        "predicted_reward_mean": float(np.mean(predicted)),
        "target_reward_mean": float(np.mean(target)),
        "target_reward_std": float(np.std(target)),
    }


def improvements(baseline: dict[str, float], candidate: dict[str, float]) -> dict[str, float]:
    lower_is_better = (
        "reward_mae",
        "reward_rmse",
        "reward_mae_normalized_by_gt_std",
        "delta_mae",
        "terminal_gain_mae",
    )
    result = {}
    for key in lower_is_better:
        denominator = max(abs(baseline[key]), 1e-12)
        result[f"{key}_improvement_percent"] = 100 * (baseline[key] - candidate[key]) / denominator
    for key in (
        "flattened_pearson",
        "mean_per_window_pearson",
        "delta_sign_agreement",
        "terminal_gain_sign_agreement",
    ):
        result[f"{key}_absolute_gain"] = candidate[key] - baseline[key]
    result["dynamic_range_ratio_distance_improvement"] = abs(baseline["dynamic_range_ratio"] - 1) - abs(candidate["dynamic_range_ratio"] - 1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--reuse-baseline-cache", help="Common arrays for a candidate-only sweep cache")
    parser.add_argument("--reuse-baseline-report", help="Previously scored report for context/target/baseline arrays")
    parser.add_argument("--reward-checkpoint", required=True)
    parser.add_argument("--t5-model", required=True)
    parser.add_argument("--reset-manifest", required=True)
    parser.add_argument("--instruction-map", help="Optional JSON seed-to-exact-instruction mapping")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompts-per-arm", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.prompts_per_arm < 1:
        raise SystemExit("--prompts-per-arm must be positive")

    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel

    with np.load(args.cache, allow_pickle=False) as values:
        candidate = values["candidate"].copy()
        candidate_paths = values["path"].astype(str).tolist()
        source_path = args.reuse_baseline_cache or args.cache
    with np.load(source_path, allow_pickle=False) as values:
        context = values["context_last"].copy()
        target = values["target"].copy()
        baseline = values["baseline"].copy()
        arm_right = values["arm_right"].astype(bool)
        capture_success = values["capture_success"].astype(bool)
        paths = values["path"].astype(str).tolist()
        synthetic_seed = values["synthetic_seed"].astype(np.int64)
        is_synthetic = synthetic_seed >= 0
        exact_instructions = values["instruction"].astype(str).tolist() if "instruction" in values.files else None
    if candidate_paths != paths:
        raise ValueError("candidate cache and reused baseline cache have different paths")

    device = torch.device(args.device)
    model = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        args.reward_checkpoint, config={"t5_model_name": args.t5_model}
    ).to(device).eval().requires_grad_(False)
    left_prompts = select_prompts(Path(args.reset_manifest), False, args.prompts_per_arm)
    right_prompts = select_prompts(Path(args.reset_manifest), True, args.prompts_per_arm)
    if exact_instructions is not None:
        if any(not value for value in exact_instructions):
            raise ValueError("cache contains an empty exact instruction")
        prompt_sets = [[value] for value in exact_instructions]
        prompt_description = "exact public-trajectory instruction stored per sequence"
    elif args.instruction_map:
        instruction_document = json.loads(Path(args.instruction_map).read_text())
        raw_mapping = instruction_document.get("seed_to_instruction", instruction_document)
        exact_mapping = {int(seed): str(instruction) for seed, instruction in raw_mapping.items()}
        missing = sorted(set(synthetic_seed.tolist()).difference(exact_mapping))
        if missing:
            raise ValueError(f"instruction map is missing synthetic seeds: {missing}")
        prompt_sets = [[exact_mapping[int(seed)]] for seed in synthetic_seed]
        prompt_description = "exact deterministic task instruction recovered from the same RoboTwin seed"
    else:
        prompt_sets = [right_prompts if value else left_prompts for value in arm_right]
        prompt_description = "mean score over deterministic arm-consistent prompts from the official reset corpus; identical prompts for GT/baseline/candidate"
    all_frames = {
        "context": context[:, None],
        "target": target,
        "baseline": baseline,
        "candidate": candidate,
    }
    scores = {}
    if args.reuse_baseline_report:
        baseline_report = json.loads(Path(args.reuse_baseline_report).read_text())
        if baseline_report["paths"] != paths:
            raise ValueError("baseline reward report and cache paths are not aligned")
        scores.update({
            name: np.asarray(baseline_report["raw_scores"][name], dtype=np.float64)
            for name in ("context", "target", "baseline")
        })
    for name, frames in all_frames.items():
        if name in scores:
            pass
        elif name == "candidate" and np.array_equal(frames, all_frames["baseline"]):
            scores[name] = scores["baseline"].copy()
        else:
            scores[name] = score_sequences(model, frames, prompt_sets, device, args.batch_size)
        print(json.dumps({"scored": name, "shape": list(scores[name].shape)}), flush=True)

    groups = {
        "overall": np.ones(len(arm_right), dtype=bool),
        "left": ~arm_right,
        "right": arm_right,
        "capture_success": capture_success,
        "capture_failure": ~capture_success,
        "official": ~is_synthetic,
        "synthetic": is_synthetic,
    }
    target_float = target.astype(np.float32) / 255.0
    previous = np.concatenate((context[:, None].astype(np.float32) / 255.0, target_float[:, :-1]), axis=1)
    motion_score = np.mean(np.abs(target_float - previous), axis=(1, 2, 3, 4))
    high_motion = motion_score >= 0.03
    groups.update({
        "low_motion": ~high_motion,
        "high_motion": high_motion,
        "left_low_motion": (~arm_right) & (~high_motion),
        "left_high_motion": (~arm_right) & high_motion,
        "right_low_motion": arm_right & (~high_motion),
        "right_high_motion": arm_right & high_motion,
    })
    group_results = {}
    for name, mask in groups.items():
        if not mask.any():
            continue
        base_metrics = metrics(scores["baseline"][mask], scores["target"][mask], scores["context"][mask, 0])
        candidate_metrics = metrics(scores["candidate"][mask], scores["target"][mask], scores["context"][mask, 0])
        group_results[name] = {
            "windows": int(mask.sum()),
            "baseline": base_metrics,
            "candidate": candidate_metrics,
            "improvement": improvements(base_metrics, candidate_metrics),
        }

    report = {
        "format": "strict-track2-frozen-official-reward-alignment-audit-v1",
        "cache": str(Path(args.cache).resolve()),
        "reuse_baseline_cache": str(Path(args.reuse_baseline_cache).resolve()) if args.reuse_baseline_cache else None,
        "reuse_baseline_report": str(Path(args.reuse_baseline_report).resolve()) if args.reuse_baseline_report else None,
        "reward_checkpoint": str(Path(args.reward_checkpoint).resolve()),
        "t5_model": str(Path(args.t5_model).resolve()),
        "prompt_protocol": {
            "description": prompt_description,
            "instruction_map": str(Path(args.instruction_map).resolve()) if args.instruction_map else None,
            "left": left_prompts,
            "right": right_prompts,
        },
        "window_count": len(paths),
        "motion_threshold": 0.03,
        "motion_score": motion_score.tolist(),
        "paths": paths,
        "groups": group_results,
        "raw_scores": {name: value.tolist() for name, value in scores.items()},
        "interpretation_guard": "This audits the frozen official reward model and does not modify it or constitute real RoboTwin policy success.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(group_results, indent=2))


if __name__ == "__main__":
    main()
