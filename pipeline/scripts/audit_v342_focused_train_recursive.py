#!/usr/bin/env python3
"""Focused public-train recursive gate for the 0.90-strength v342 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from audit_v310_full_mirror_causal_gate import score_terminal, sha256
from wam_pipeline.v342_temporal_blended_public_reanchor_runtime import (
    Track2V342TemporalBlendedPublicReanchor,
)


ALIGNMENTS = (0, 4)


def seed_for(path: Path) -> int:
    return int.from_bytes(hashlib.sha256(path.name.encode()).digest()[:8], "little") % (2**31)


def rgb_mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.abs(prediction.astype(np.int16) - target.astype(np.int16)).mean())


def temporal_error(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.abs(
        np.diff(prediction.astype(np.float32), axis=0)
        - np.diff(target.astype(np.float32), axis=0)
    ).mean())


def run_candidate(args, episodes: list[int], instructions: dict[str, str], reward) -> list[dict]:
    runtime = Track2V342TemporalBlendedPublicReanchor(
        args.checkpoint_dir, args.library_index, args.device,
        args.action_gate, args.phase_gate, args.recursive_ood_gate,
        feature_workers=args.feature_workers,
    )
    states = []
    for episode in episodes:
        available = {
            int(path.stem.split("_")[1]): path
            for path in args.windows.glob(f"episode{episode}_*.npz")
        }
        for alignment in ALIGNMENTS:
            if alignment not in available:
                continue
            with np.load(available[alignment], allow_pickle=False) as payload:
                initial = payload["context_frames"].astype(np.uint8)
            states.append({
                "episode": episode, "alignment": alignment, "start": alignment,
                "available": available, "recursive_context": initial,
                "instruction": str(instructions[str(episode)]),
            })
    rows = []
    reward_frames = []
    reward_prompts = []
    while True:
        active = [state for state in states if state["start"] in state["available"]]
        if not active:
            break
        requests = []
        pair_contexts = []
        pair_histories = []
        pair_futures = []
        pair_seeds = []
        pair_prompts = []
        for state in active:
            path = state["available"][state["start"]]
            with np.load(path, allow_pickle=False) as payload:
                real = payload["context_frames"].astype(np.uint8)
                history = payload["history_actions"].astype(np.float32)
                future = payload["future_actions"].astype(np.float32)
                target = payload["target_frames"].astype(np.uint8)
            teacher_reanchor, teacher_probability = runtime._recursive_reanchor_gate(
                real, history, future
            )
            requests.append({
                "state": state, "path": path, "real": real, "history": history,
                "future": future, "target": target,
                "teacher_reanchor": teacher_reanchor,
                "teacher_probability": teacher_probability,
            })
            for context in (real, state["recursive_context"]):
                pair_contexts.append(context)
                pair_histories.append(history)
                pair_futures.append(future)
                pair_seeds.append(seed_for(path))
                pair_prompts.append(state["instruction"])
        predictions = []
        masks = []
        probabilities = []
        rows_selected = []
        for begin in range(0, len(pair_contexts), args.inference_batch_size):
            end = begin + args.inference_batch_size
            prediction = runtime.predict_batch(
                np.stack(pair_contexts[begin:end]), np.stack(pair_histories[begin:end]),
                np.stack(pair_futures[begin:end]), np.asarray(pair_seeds[begin:end], dtype=np.int64),
                pair_prompts[begin:end],
            )
            predictions.extend(prediction)
            masks.extend(runtime.last_recursive_reanchor_mask.tolist())
            probabilities.extend(runtime.last_recursive_ood_probabilities.tolist())
            rows_selected.extend(runtime.last_recursive_reanchor_rows.tolist())
        for index, request in enumerate(requests):
            teacher = predictions[2 * index]
            recursive = predictions[2 * index + 1]
            state = request["state"]
            target_context = request["target"][-5:]
            reward_offset = len(reward_frames)
            reward_frames.extend((request["real"][-1], request["target"][-1], teacher[-1], recursive[-1]))
            reward_prompts.extend([state["instruction"]] * 4)
            rows.append({
                "key": f"episode{state['episode']}/a{state['alignment']}/s{state['start']:05d}",
                "episode": state["episode"], "alignment": state["alignment"], "start": state["start"],
                "teacher_reanchor_precheck": request["teacher_reanchor"],
                "teacher_reanchor_runtime": bool(masks[2 * index]),
                "recursive_reanchor": bool(masks[2 * index + 1]),
                "teacher_ood_probability": request["teacher_probability"],
                "recursive_ood_probability": float(probabilities[2 * index + 1]),
                "recursive_reanchor_row": int(rows_selected[2 * index + 1]),
                "teacher_rgb_mae": rgb_mae(teacher[-5:], target_context),
                "recursive_rgb_mae": rgb_mae(recursive[-5:], target_context),
                "teacher_temporal_error": temporal_error(teacher[-5:], target_context),
                "recursive_temporal_error": temporal_error(recursive[-5:], target_context),
                "reward_offset": reward_offset,
            })
            state["recursive_context"] = recursive[-5:].copy()
            state["start"] += 8
    scores = score_terminal(
        reward, np.stack(reward_frames), reward_prompts,
        torch.device(args.device), args.reward_batch_size,
    )
    for row in rows:
        offset = row.pop("reward_offset")
        _, gt_reward, teacher_reward, recursive_reward = scores[offset : offset + 4]
        row.update({
            "gt_reward": float(gt_reward), "teacher_reward": float(teacher_reward),
            "recursive_reward": float(recursive_reward),
            "teacher_reward_error": float(abs(teacher_reward - gt_reward)),
            "recursive_reward_error": float(abs(recursive_reward - gt_reward)),
        })
    print(f"FOCUSED_TRAIN_MODEL_SCORED v342 {len(rows)}", flush=True)
    return rows


def stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "mean": float(array.mean()), "median": float(np.median(array))}


def aggregate(rows: list[dict], keys: set[str]) -> dict:
    result = {}
    for group, selected in (("all", rows), ("direct", [row for row in rows if row["key"] in keys])):
        result[group] = {
            metric: stats([float(row[metric]) for row in selected])
            for metric in (
                "teacher_rgb_mae", "recursive_rgb_mae", "teacher_temporal_error",
                "recursive_temporal_error", "teacher_reward_error",
                "recursive_reward_error", "teacher_reward", "recursive_reward",
            )
        }
        result[group]["recursive_reward_hit_rate_at_0p9"] = float(
            np.mean([row["recursive_reward"] >= 0.90 for row in selected])
        ) if selected else None
    return result


def ratio(a: float, b: float) -> float:
    return float(a / max(b, 1e-9))


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "checkpoint-dir", "library-index", "action-gate", "phase-gate", "recursive-ood-gate",
        "windows", "instruction-map", "reward-checkpoint", "t5-model",
        "v340-train-report", "preregistration", "output",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--reward-batch-size", type=int, default=32)
    parser.add_argument("--feature-workers", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prereg = json.loads(args.preregistration.read_text())
    if prereg.get("format") != "strict-track2-v342-focused-train-recursive-preregistration-v1":
        raise RuntimeError("wrong v342 preregistration")
    baseline_report = json.loads(args.v340_train_report.read_text())
    if baseline_report.get("passed") is not True:
        raise RuntimeError("v340 train gate failed")
    mapping = json.loads(args.instruction_map.read_text())
    episodes = [
        int(episode) for episode in mapping["train_episodes"]
        if mapping["arm_by_episode"][str(episode)] == "right"
    ]
    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel
    reward = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        str(args.reward_checkpoint), config={"t5_model_name": str(args.t5_model)}
    ).to(args.device).eval().requires_grad_(False)
    candidate = run_candidate(args, episodes, mapping["episode_to_instruction"], reward)
    baseline = [row for row in baseline_report["rows"]["v334"] if row["alignment"] in ALIGNMENTS]
    if [row["key"] for row in baseline] != [row["key"] for row in candidate]:
        raise RuntimeError("focused baseline/candidate rows misaligned")
    direct = {row["key"] for row in candidate if row["recursive_reanchor"]}
    aggregates = {"v334": aggregate(baseline, direct), "v342": aggregate(candidate, direct)}
    comparisons = {
        group: {
            metric: ratio(aggregates["v342"][group][metric]["mean"], aggregates["v334"][group][metric]["mean"])
            for metric in ("recursive_rgb_mae", "recursive_temporal_error", "recursive_reward_error")
        }
        for group in ("all", "direct")
    }
    teacher_rgb_delta = max(abs(a["teacher_rgb_mae"] - b["teacher_rgb_mae"]) for a, b in zip(candidate, baseline, strict=True))
    teacher_temporal_delta = max(abs(a["teacher_temporal_error"] - b["teacher_temporal_error"]) for a, b in zip(candidate, baseline, strict=True))
    checks = {
        "exact_484_rows": len(candidate) == 484,
        "teacher_precheck_reanchors_zero": not any(row["teacher_reanchor_precheck"] for row in candidate),
        "teacher_runtime_reanchors_zero": not any(row["teacher_reanchor_runtime"] for row in candidate),
        "teacher_rgb_metric_max_delta_le_0p01": teacher_rgb_delta <= 0.01,
        "teacher_temporal_metric_max_delta_le_0p01": teacher_temporal_delta <= 0.01,
        "direct_reanchors_ge_20": len(direct) >= 20,
        "direct_rgb_ratio_le_0p90": comparisons["direct"]["recursive_rgb_mae"] <= 0.90,
        "direct_temporal_ratio_le_0p90": comparisons["direct"]["recursive_temporal_error"] <= 0.90,
        "direct_reward_error_ratio_le_1p10": comparisons["direct"]["recursive_reward_error"] <= 1.10,
        "direct_reward_mean_ge_0p90": aggregates["v342"]["direct"]["recursive_reward"]["mean"] >= 0.90,
        "direct_reward_hit_rate_ge_0p85": aggregates["v342"]["direct"]["recursive_reward_hit_rate_at_0p9"] >= 0.85,
        "all_rgb_ratio_le_1p02": comparisons["all"]["recursive_rgb_mae"] <= 1.02,
        "all_temporal_ratio_le_1p02": comparisons["all"]["recursive_temporal_error"] <= 1.02,
        "all_reward_error_ratio_le_1p02": comparisons["all"]["recursive_reward_error"] <= 1.02,
    }
    report = {
        "format": "strict-track2-v342-focused-train-recursive-gate-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": "track2-v342-temporal-blended-public-reanchor-v339-v334",
        "alignments": list(ALIGNMENTS), "rows_found": len(candidate),
        "direct_reanchor_count": len(direct),
        "teacher_metric_max_delta": {"rgb": teacher_rgb_delta, "temporal": teacher_temporal_delta},
        "aggregates": aggregates, "v342_over_v334": comparisons,
        "checks": checks, "passed": all(checks.values()),
        "authorizes_public_holdout_gate": all(checks.values()),
        "evidence_sha256": {
            "preregistration": sha256(args.preregistration),
            "v340_train_report": sha256(args.v340_train_report),
            "recursive_ood_gate": sha256(args.recursive_ood_gate),
            "reward_checkpoint": sha256(args.reward_checkpoint),
        },
        "guards": {
            "public_train_only": True, "official_batch16_outcomes_read": False,
            "hidden_or_final_data": False, "real_submission": False,
        },
        "rows": {"v342": candidate},
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "direct_reanchor_count": len(direct), "teacher_metric_max_delta": report["teacher_metric_max_delta"],
        "v342_over_v334": comparisons, "checks": checks, "passed": report["passed"],
    }, indent=2), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
