#!/usr/bin/env python3
"""All-offset public-holdout recursive audit for frozen v340 semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from audit_v310_full_mirror_causal_gate import RIGHT_EPISODES, score_terminal, sha256
from wam_pipeline.v340_high_specificity_public_reanchor_runtime import (
    Track2V340HighSpecificityPublicReanchor,
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


def digest(frames: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(frames).tobytes()).hexdigest()


def run_candidate(args, instructions: dict[str, str], reward) -> list[dict]:
    runtime = Track2V340HighSpecificityPublicReanchor(
        args.checkpoint_dir, args.library_index, args.device,
        args.action_gate, args.phase_gate, args.recursive_ood_gate,
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
            recursive_reanchor, recursive_ood_probability = runtime._recursive_reanchor_gate(
                state["recursive_context"], history, future
            )
            requests.append({
                "state": state, "path": path, "real": real, "history": history,
                "future": future, "target": target,
                "recursive_reanchor": recursive_reanchor,
                "recursive_ood_probability": recursive_ood_probability,
            })
            for context in (real, state["recursive_context"]):
                pair_contexts.append(context)
                pair_histories.append(history)
                pair_futures.append(future)
                pair_seeds.append(seed_for(path))
                pair_prompts.append(state["instruction"])
        predictions = []
        for begin in range(0, len(pair_contexts), args.inference_batch_size):
            end = begin + args.inference_batch_size
            predictions.extend(runtime.predict_batch(
                np.stack(pair_contexts[begin:end]), np.stack(pair_histories[begin:end]),
                np.stack(pair_futures[begin:end]), np.asarray(pair_seeds[begin:end], dtype=np.int64),
                pair_prompts[begin:end],
            ))
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
                "recursive_reanchor": request["recursive_reanchor"],
                "recursive_ood_probability": request["recursive_ood_probability"],
                "teacher_next_context_rgb_mae": rgb_mae(teacher[-5:], target_context),
                "recursive_next_context_rgb_mae": rgb_mae(recursive[-5:], target_context),
                "teacher_temporal_delta_error": temporal_error(teacher[-5:], target_context),
                "recursive_temporal_delta_error": temporal_error(recursive[-5:], target_context),
                "teacher_next_context_sha256": digest(teacher[-5:]),
                "recursive_next_context_sha256": digest(recursive[-5:]),
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
    print(f"HOLDOUT_RECURSIVE_MODEL_SCORED v340 {len(rows)}", flush=True)
    return rows


def stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values), "mean": float(array.mean()),
        "median": float(np.median(array)), "p90": float(np.quantile(array, 0.9)),
    }


def aggregate(rows: list[dict], direct_keys: set[str]) -> dict:
    result = {}
    for split in RIGHT_EPISODES:
        split_rows = [row for row in rows if row["split"] == split]
        result[split] = {}
        for group, selected in (
            ("all", split_rows),
            ("direct_reanchor", [row for row in split_rows if row["key"] in direct_keys]),
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


def ratio(candidate: float, baseline: float) -> float:
    return float(candidate / max(baseline, 1e-9))


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "checkpoint-dir", "library-index", "action-gate", "phase-gate", "recursive-ood-gate",
        "windows", "instruction-map", "reward-checkpoint", "t5-model",
        "v335-baseline-report", "v335-causal-report", "v340-train-report",
        "preregistration", "output",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--reward-batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prereg = json.loads(args.preregistration.read_text())
    if prereg.get("format") != "strict-track2-v341-v340-public-holdout-recursive-preregistration-v1":
        raise RuntimeError("wrong v341 preregistration")
    train = json.loads(args.v340_train_report.read_text())
    if train.get("passed") is not True:
        raise RuntimeError("v340 train recursive gate failed")
    baseline_report = json.loads(args.v335_baseline_report.read_text())
    if baseline_report.get("format") != "strict-track2-v335-all-offset-recursive-stability-gate-v1":
        raise RuntimeError("wrong frozen v334/v335 baseline report")
    causal = json.loads(args.v335_causal_report.read_text())
    if causal.get("passed") is not True:
        raise RuntimeError("v334/v335 teacher causal gate failed")
    mapping = json.loads(args.instruction_map.read_text())
    if mapping.get("hidden_or_final_evaluation_data") is not False:
        raise RuntimeError("instruction map boundary invalid")
    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel
    reward = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        str(args.reward_checkpoint), config={"t5_model_name": str(args.t5_model)}
    ).to(args.device).eval().requires_grad_(False)
    candidate = run_candidate(args, mapping["episode_to_instruction"], reward)
    baseline = baseline_report["rows"]["v335"]
    if [row["key"] for row in baseline] != [row["key"] for row in candidate]:
        raise RuntimeError("baseline/candidate holdout rows misaligned")
    direct_keys = {row["key"] for row in candidate if row["recursive_reanchor"]}
    aggregates = {
        "v334": aggregate(baseline, direct_keys),
        "v340": aggregate(candidate, direct_keys),
    }
    comparisons = {}
    for split in RIGHT_EPISODES:
        comparisons[split] = {}
        for group in ("all", "direct_reanchor"):
            comparisons[split][group] = {
                metric: ratio(
                    aggregates["v340"][split][group][metric]["mean"],
                    aggregates["v334"][split][group][metric]["mean"],
                )
                for metric in (
                    "recursive_next_context_rgb_mae", "recursive_temporal_delta_error",
                    "recursive_reward_absolute_error",
                )
            }
    teacher_exact = all(
        base["teacher_next_context_sha256"] == cand["teacher_next_context_sha256"]
        for base, cand in zip(baseline, candidate, strict=True)
    )
    counts = {
        split: sum(row["recursive_reanchor"] and row["split"] == split for row in candidate)
        for split in RIGHT_EPISODES
    }
    checks = {
        "v340_train_gate": True,
        "v334_teacher_causal_gate": True,
        "exact_512_rows": len(candidate) == 512,
        "teacher_next_context_bit_exact_v334": teacher_exact,
        "validation_direct_reanchors_ge_10": counts["validation"] >= 10,
        "local_direct_reanchors_ge_10": counts["local_test"] >= 10,
    }
    for split in RIGHT_EPISODES:
        prefix = "validation" if split == "validation" else "local"
        checks.update({
            f"{prefix}_direct_rgb_ratio_le_0p90": comparisons[split]["direct_reanchor"]["recursive_next_context_rgb_mae"] <= 0.90,
            f"{prefix}_direct_temporal_ratio_le_0p90": comparisons[split]["direct_reanchor"]["recursive_temporal_delta_error"] <= 0.90,
            f"{prefix}_direct_reward_error_ratio_le_1p10": comparisons[split]["direct_reanchor"]["recursive_reward_absolute_error"] <= 1.10,
            f"{prefix}_direct_reward_mean_ge_0p90": aggregates["v340"][split]["direct_reanchor"]["recursive_terminal_reward"]["mean"] >= 0.90,
            f"{prefix}_direct_reward_hit_rate_ge_0p85": aggregates["v340"][split]["direct_reanchor"]["recursive_reward_hit_rate_at_0p9"] >= 0.85,
            f"{prefix}_all_rgb_ratio_le_1p02": comparisons[split]["all"]["recursive_next_context_rgb_mae"] <= 1.02,
            f"{prefix}_all_temporal_ratio_le_1p02": comparisons[split]["all"]["recursive_temporal_delta_error"] <= 1.02,
            f"{prefix}_all_reward_error_ratio_le_1p02": comparisons[split]["all"]["recursive_reward_absolute_error"] <= 1.02,
        })
    report = {
        "format": "strict-track2-v341-v340-public-holdout-recursive-gate-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": "track2-v340-high-specificity-public-reanchor-v339-v334",
        "direct_reanchor_counts": counts,
        "aggregates": aggregates,
        "v340_over_v334": comparisons,
        "teacher_next_context_bit_exact": teacher_exact,
        "checks": checks,
        "passed": all(checks.values()),
        "authorizes_service_acceptance_only": all(checks.values()),
        "evidence_sha256": {
            "preregistration": sha256(args.preregistration),
            "v340_train_report": sha256(args.v340_train_report),
            "v335_baseline_report": sha256(args.v335_baseline_report),
            "v335_causal_report": sha256(args.v335_causal_report),
            "recursive_ood_gate": sha256(args.recursive_ood_gate),
            "reward_checkpoint": sha256(args.reward_checkpoint),
        },
        "guards": {
            "public_world_model_holdouts_only": True,
            "official_batch16_outcomes_read": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
        "rows": {"v340": candidate},
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "direct_reanchor_counts": counts, "v340_over_v334": comparisons,
        "checks": checks, "passed": report["passed"],
    }, indent=2), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
