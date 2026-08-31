#!/usr/bin/env python3
"""Final all-offset public-WM holdout gate for frozen v342 semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from audit_v310_full_mirror_causal_gate import RIGHT_EPISODES, score_terminal, sha256
from wam_pipeline.v342_temporal_blended_public_reanchor_runtime import (
    Track2V342TemporalBlendedPublicReanchor,
)


def seed_for(path: Path) -> int:
    return int.from_bytes(hashlib.sha256(path.name.encode()).digest()[:8], "little") % (2**31)


def rgb_mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.abs(prediction.astype(np.int16) - target.astype(np.int16)).mean())


def temporal_error(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.abs(
        np.diff(prediction.astype(np.float32), axis=0)
        - np.diff(target.astype(np.float32), axis=0)
    ).mean())


def run_candidate(args, instructions: dict[str, str], reward) -> list[dict]:
    runtime = Track2V342TemporalBlendedPublicReanchor(
        args.checkpoint_dir, args.library_index, args.device,
        args.action_gate, args.phase_gate, args.recursive_ood_gate,
        feature_workers=args.feature_workers,
    )
    states = []
    for split, episodes in RIGHT_EPISODES.items():
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
                    "split": split, "episode": episode, "alignment": alignment,
                    "start": alignment, "available": available,
                    "recursive_context": initial,
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
        contexts = []
        histories = []
        futures = []
        seeds = []
        prompts = []
        for state in active:
            path = state["available"][state["start"]]
            with np.load(path, allow_pickle=False) as payload:
                real = payload["context_frames"].astype(np.uint8)
                history = payload["history_actions"].astype(np.float32)
                future = payload["future_actions"].astype(np.float32)
                target = payload["target_frames"].astype(np.uint8)
            requests.append({
                "state": state, "path": path, "real": real,
                "history": history, "future": future, "target": target,
            })
            for context in (real, state["recursive_context"]):
                contexts.append(context); histories.append(history); futures.append(future)
                seeds.append(seed_for(path)); prompts.append(state["instruction"])
        predictions = []
        masks = []
        probabilities = []
        selected_rows = []
        for begin in range(0, len(contexts), args.inference_batch_size):
            end = begin + args.inference_batch_size
            prediction = runtime.predict_batch(
                np.stack(contexts[begin:end]), np.stack(histories[begin:end]),
                np.stack(futures[begin:end]), np.asarray(seeds[begin:end], dtype=np.int64),
                prompts[begin:end],
            )
            predictions.extend(prediction)
            masks.extend(runtime.last_recursive_reanchor_mask.tolist())
            probabilities.extend(runtime.last_recursive_ood_probabilities.tolist())
            selected_rows.extend(runtime.last_recursive_reanchor_rows.tolist())
        for index, request in enumerate(requests):
            teacher = predictions[2 * index]
            recursive = predictions[2 * index + 1]
            state = request["state"]
            target_context = request["target"][-5:]
            reward_offset = len(reward_frames)
            reward_frames.extend((request["real"][-1], request["target"][-1], teacher[-1], recursive[-1]))
            reward_prompts.extend([state["instruction"]] * 4)
            rows.append({
                "key": f"{state['split']}/episode{state['episode']}/a{state['alignment']}/s{state['start']:05d}",
                "split": state["split"], "episode": state["episode"],
                "alignment": state["alignment"], "start": state["start"],
                "teacher_reanchor": bool(masks[2 * index]),
                "recursive_reanchor": bool(masks[2 * index + 1]),
                "teacher_ood_probability": float(probabilities[2 * index]),
                "recursive_ood_probability": float(probabilities[2 * index + 1]),
                "recursive_reanchor_row": int(selected_rows[2 * index + 1]),
                "teacher_next_context_rgb_mae": rgb_mae(teacher[-5:], target_context),
                "recursive_next_context_rgb_mae": rgb_mae(recursive[-5:], target_context),
                "teacher_temporal_delta_error": temporal_error(teacher[-5:], target_context),
                "recursive_temporal_delta_error": temporal_error(recursive[-5:], target_context),
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
        context_reward, gt_reward, teacher_reward, recursive_reward = scores[offset : offset + 4]
        row.update({
            "context_terminal_reward": float(context_reward),
            "gt_terminal_reward": float(gt_reward),
            "teacher_terminal_reward": float(teacher_reward),
            "recursive_terminal_reward": float(recursive_reward),
            "teacher_reward_absolute_error": float(abs(teacher_reward - gt_reward)),
            "recursive_reward_absolute_error": float(abs(recursive_reward - gt_reward)),
        })
    print(f"HOLDOUT_RECURSIVE_MODEL_SCORED v342 {len(rows)}", flush=True)
    return rows


def stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "mean": float(array.mean()), "median": float(np.median(array))}


def aggregate(rows: list[dict], direct_keys: set[str]) -> dict:
    result = {}
    for split in RIGHT_EPISODES:
        split_rows = [row for row in rows if row["split"] == split]
        result[split] = {}
        for group, selected in (
            ("all", split_rows),
            ("direct", [row for row in split_rows if row["key"] in direct_keys]),
        ):
            result[split][group] = {
                metric: stats([float(row[metric]) for row in selected])
                for metric in (
                    "teacher_next_context_rgb_mae", "recursive_next_context_rgb_mae",
                    "teacher_temporal_delta_error", "recursive_temporal_delta_error",
                    "teacher_reward_absolute_error", "recursive_reward_absolute_error",
                    "teacher_terminal_reward", "recursive_terminal_reward",
                )
            }
            result[split][group]["recursive_reward_hit_rate_at_0p9"] = float(
                np.mean([row["recursive_terminal_reward"] >= 0.90 for row in selected])
            ) if selected else None
    return result


def ratio(a: float, b: float) -> float:
    return float(a / max(b, 1e-9))


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "checkpoint-dir", "library-index", "action-gate", "phase-gate", "recursive-ood-gate",
        "windows", "instruction-map", "reward-checkpoint", "t5-model",
        "v335-baseline-report", "v335-causal-report", "v342-train-report",
        "preregistration", "output",
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
    if prereg.get("format") != "strict-track2-v343-v342-public-holdout-recursive-preregistration-v1":
        raise RuntimeError("wrong v343 preregistration")
    focused = json.loads(args.v342_train_report.read_text())
    if focused.get("passed") is not True:
        raise RuntimeError("v342 focused train gate failed")
    causal = json.loads(args.v335_causal_report.read_text())
    if causal.get("passed") is not True:
        raise RuntimeError("v334 teacher causal gate failed")
    baseline_report = json.loads(args.v335_baseline_report.read_text())
    baseline = baseline_report["rows"]["v335"]
    mapping = json.loads(args.instruction_map.read_text())
    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel
    reward = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        str(args.reward_checkpoint), config={"t5_model_name": str(args.t5_model)}
    ).to(args.device).eval().requires_grad_(False)
    candidate = run_candidate(args, mapping["episode_to_instruction"], reward)
    if [row["key"] for row in baseline] != [row["key"] for row in candidate]:
        raise RuntimeError("holdout baseline/candidate rows misaligned")
    direct = {row["key"] for row in candidate if row["recursive_reanchor"]}
    aggregates = {"v334": aggregate(baseline, direct), "v342": aggregate(candidate, direct)}
    comparisons = {}
    for split in RIGHT_EPISODES:
        comparisons[split] = {
            group: {
                metric: ratio(
                    aggregates["v342"][split][group][metric]["mean"],
                    aggregates["v334"][split][group][metric]["mean"],
                )
                for metric in (
                    "recursive_next_context_rgb_mae", "recursive_temporal_delta_error",
                    "recursive_reward_absolute_error",
                )
            }
            for group in ("all", "direct")
        }
    counts = {
        split: sum(row["recursive_reanchor"] and row["split"] == split for row in candidate)
        for split in RIGHT_EPISODES
    }
    teacher_rgb_delta = max(abs(a["teacher_next_context_rgb_mae"] - b["teacher_next_context_rgb_mae"]) for a, b in zip(candidate, baseline, strict=True))
    teacher_temporal_delta = max(abs(a["teacher_temporal_delta_error"] - b["teacher_temporal_delta_error"]) for a, b in zip(candidate, baseline, strict=True))
    checks = {
        "v342_focused_train_gate": True,
        "v334_teacher_causal_gate": True,
        "exact_512_rows": len(candidate) == 512,
        "teacher_reanchor_count_zero": not any(row["teacher_reanchor"] for row in candidate),
        "teacher_rgb_metric_max_delta_le_0p01": teacher_rgb_delta <= 0.01,
        "teacher_temporal_metric_max_delta_le_0p01": teacher_temporal_delta <= 0.01,
        "validation_direct_reanchors_ge_10": counts["validation"] >= 10,
        "local_direct_reanchors_ge_10": counts["local_test"] >= 10,
    }
    for split in RIGHT_EPISODES:
        prefix = "validation" if split == "validation" else "local"
        checks.update({
            f"{prefix}_direct_rgb_ratio_le_0p90": comparisons[split]["direct"]["recursive_next_context_rgb_mae"] <= 0.90,
            f"{prefix}_direct_temporal_ratio_le_0p90": comparisons[split]["direct"]["recursive_temporal_delta_error"] <= 0.90,
            f"{prefix}_direct_reward_error_ratio_le_1p10": comparisons[split]["direct"]["recursive_reward_absolute_error"] <= 1.10,
            f"{prefix}_direct_reward_mean_ge_0p90": aggregates["v342"][split]["direct"]["recursive_terminal_reward"]["mean"] >= 0.90,
            f"{prefix}_direct_reward_hit_rate_ge_0p85": aggregates["v342"][split]["direct"]["recursive_reward_hit_rate_at_0p9"] >= 0.85,
            f"{prefix}_all_rgb_ratio_le_1p02": comparisons[split]["all"]["recursive_next_context_rgb_mae"] <= 1.02,
            f"{prefix}_all_temporal_ratio_le_1p02": comparisons[split]["all"]["recursive_temporal_delta_error"] <= 1.02,
            f"{prefix}_all_reward_error_ratio_le_1p02": comparisons[split]["all"]["recursive_reward_absolute_error"] <= 1.02,
        })
    report = {
        "format": "strict-track2-v343-v342-public-holdout-recursive-gate-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": "track2-v342-temporal-blended-public-reanchor-v339-v334",
        "direct_reanchor_counts": counts,
        "teacher_metric_max_delta": {"rgb": teacher_rgb_delta, "temporal": teacher_temporal_delta},
        "aggregates": aggregates, "v342_over_v334": comparisons,
        "checks": checks, "passed": all(checks.values()),
        "authorizes_service_acceptance_only": all(checks.values()),
        "evidence_sha256": {
            "preregistration": sha256(args.preregistration),
            "v342_train_report": sha256(args.v342_train_report),
            "v335_baseline_report": sha256(args.v335_baseline_report),
            "v335_causal_report": sha256(args.v335_causal_report),
            "recursive_ood_gate": sha256(args.recursive_ood_gate),
            "reward_checkpoint": sha256(args.reward_checkpoint),
        },
        "guards": {
            "public_world_model_holdout_only": True,
            "official_batch16_outcomes_read": False,
            "hidden_or_final_data": False, "real_submission": False,
        },
        "rows": {"v342": candidate},
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "direct_reanchor_counts": counts, "teacher_metric_max_delta": report["teacher_metric_max_delta"],
        "v342_over_v334": comparisons, "checks": checks, "passed": report["passed"],
    }, indent=2), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
