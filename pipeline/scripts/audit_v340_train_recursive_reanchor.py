#!/usr/bin/env python3
"""All-offset public-train recursive gate for the frozen v340 reanchor."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from audit_v310_full_mirror_causal_gate import score_terminal, sha256
from wam_pipeline.v334_bounded_onset_terminal_runtime import Track2V334BoundedOnsetTerminal
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


def instantiate(name: str, args):
    if name == "v334":
        return Track2V334BoundedOnsetTerminal(
            args.checkpoint_dir, args.library_index, args.device,
            args.action_gate, args.phase_gate,
        )
    return Track2V340HighSpecificityPublicReanchor(
        args.checkpoint_dir, args.library_index, args.device,
        args.action_gate, args.phase_gate, args.recursive_ood_gate,
    )


def run_model(name: str, args, episodes: list[int], instructions: dict[str, str], reward) -> list[dict]:
    runtime = instantiate(name, args)
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
            teacher_reanchor = False
            recursive_reanchor = False
            teacher_ood_probability = None
            recursive_ood_probability = None
            if name == "v340":
                teacher_reanchor, teacher_ood_probability = runtime._recursive_reanchor_gate(
                    real, history, future
                )
                recursive_reanchor, recursive_ood_probability = runtime._recursive_reanchor_gate(
                    state["recursive_context"], history, future
                )
            requests.append({
                "state": state, "path": path, "real": real, "history": history,
                "future": future, "target": target,
                "teacher_reanchor": teacher_reanchor,
                "recursive_reanchor": recursive_reanchor,
                "teacher_ood_probability": teacher_ood_probability,
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
                np.stack(pair_contexts[begin:end]),
                np.stack(pair_histories[begin:end]),
                np.stack(pair_futures[begin:end]),
                np.asarray(pair_seeds[begin:end], dtype=np.int64),
                pair_prompts[begin:end],
            ))
        for index, request in enumerate(requests):
            teacher = predictions[2 * index]
            recursive = predictions[2 * index + 1]
            target_context = request["target"][-5:]
            state = request["state"]
            offset = len(reward_frames)
            reward_frames.extend((request["real"][-1], request["target"][-1], teacher[-1], recursive[-1]))
            reward_prompts.extend([state["instruction"]] * 4)
            rows.append({
                "key": f"episode{state['episode']}/a{state['alignment']}/s{state['start']:05d}",
                "episode": state["episode"], "alignment": state["alignment"], "start": state["start"],
                "teacher_reanchor": request["teacher_reanchor"],
                "recursive_reanchor": request["recursive_reanchor"],
                "teacher_ood_probability": request["teacher_ood_probability"],
                "recursive_ood_probability": request["recursive_ood_probability"],
                "teacher_rgb_mae": rgb_mae(teacher[-5:], target_context),
                "recursive_rgb_mae": rgb_mae(recursive[-5:], target_context),
                "teacher_temporal_error": temporal_error(teacher[-5:], target_context),
                "recursive_temporal_error": temporal_error(recursive[-5:], target_context),
                "teacher_sha256": digest(teacher),
                "recursive_sha256": digest(recursive),
                "reward_offset": offset,
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
            "context_reward": float(context_reward), "gt_reward": float(gt_reward),
            "teacher_reward": float(teacher_reward), "recursive_reward": float(recursive_reward),
            "teacher_reward_error": float(abs(teacher_reward - gt_reward)),
            "recursive_reward_error": float(abs(recursive_reward - gt_reward)),
        })
    del runtime
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    print(f"TRAIN_RECURSIVE_MODEL_SCORED {name} {len(rows)}", flush=True)
    return rows


def stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values), "mean": float(array.mean()),
        "median": float(np.median(array)), "p90": float(np.quantile(array, 0.9)),
    }


def aggregate(rows: list[dict], changed: set[str]) -> dict:
    result = {}
    for group, selected in (
        ("all", rows),
        ("reanchor", [row for row in rows if row["key"] in changed]),
    ):
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


def ratio(candidate: float, baseline: float) -> float:
    return float(candidate / max(baseline, 1e-9))


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "checkpoint-dir", "library-index", "action-gate", "phase-gate", "recursive-ood-gate",
        "windows", "instruction-map", "reward-checkpoint", "t5-model",
        "ood-calibration-report", "ood-holdout-report", "preregistration", "output",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--reward-batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prereg = json.loads(args.preregistration.read_text())
    if prereg.get("format") != "strict-track2-v340-train-recursive-reanchor-preregistration-v1":
        raise RuntimeError("wrong v340 preregistration")
    if json.loads(args.ood_calibration_report.read_text()).get("passed") is not True:
        raise RuntimeError("v339 calibration failed")
    if json.loads(args.ood_holdout_report.read_text()).get("passed") is not True:
        raise RuntimeError("v339 holdout failed")
    mapping = json.loads(args.instruction_map.read_text())
    if mapping.get("hidden_or_final_evaluation_data") is not False:
        raise RuntimeError("instruction map boundary is invalid")
    episodes = [
        int(episode) for episode in mapping["train_episodes"]
        if mapping["arm_by_episode"][str(episode)] == "right"
    ]
    if len(episodes) != 15:
        raise RuntimeError("expected 15 public-train right episodes")
    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel
    reward = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        str(args.reward_checkpoint), config={"t5_model_name": str(args.t5_model)}
    ).to(args.device).eval().requires_grad_(False)
    rows = {
        name: run_model(name, args, episodes, mapping["episode_to_instruction"], reward)
        for name in ("v334", "v340")
    }
    if [row["key"] for row in rows["v334"]] != [row["key"] for row in rows["v340"]]:
        raise RuntimeError("v340 rows misaligned")
    changed = {row["key"] for row in rows["v340"] if row["recursive_reanchor"]}
    aggregates = {name: aggregate(value, changed) for name, value in rows.items()}
    comparisons = {
        group: {
            metric: ratio(aggregates["v340"][group][metric]["mean"], aggregates["v334"][group][metric]["mean"])
            for metric in ("recursive_rgb_mae", "recursive_temporal_error", "recursive_reward_error")
        }
        for group in ("all", "reanchor")
    }
    teacher_exact = all(
        base["teacher_sha256"] == candidate["teacher_sha256"]
        for base, candidate in zip(rows["v334"], rows["v340"], strict=True)
    )
    checks = {
        "v339_calibration_gate": True,
        "v339_holdout_gate": True,
        "exact_1926_rows": len(rows["v340"]) == 1926,
        "teacher_reanchor_count_zero": not any(row["teacher_reanchor"] for row in rows["v340"]),
        "teacher_predictions_bit_exact_v334": teacher_exact,
        "recursive_reanchor_count_ge_100": len(changed) >= 100,
        "reanchor_rgb_ratio_le_0p90": comparisons["reanchor"]["recursive_rgb_mae"] <= 0.90,
        "reanchor_temporal_ratio_le_0p90": comparisons["reanchor"]["recursive_temporal_error"] <= 0.90,
        "reanchor_reward_error_ratio_le_1p10": comparisons["reanchor"]["recursive_reward_error"] <= 1.10,
        "reanchor_reward_mean_ge_0p90": aggregates["v340"]["reanchor"]["recursive_reward"]["mean"] >= 0.90,
        "reanchor_reward_hit_rate_ge_0p85": aggregates["v340"]["reanchor"]["recursive_reward_hit_rate_at_0p9"] >= 0.85,
        "all_rgb_ratio_le_1p02": comparisons["all"]["recursive_rgb_mae"] <= 1.02,
        "all_temporal_ratio_le_1p02": comparisons["all"]["recursive_temporal_error"] <= 1.02,
        "all_reward_error_ratio_le_1p02": comparisons["all"]["recursive_reward_error"] <= 1.02,
    }
    report = {
        "format": "strict-track2-v340-train-recursive-reanchor-gate-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": "track2-v340-high-specificity-public-reanchor-v339-v334",
        "train_episodes": episodes,
        "rows": rows,
        "recursive_reanchor_count": len(changed),
        "aggregates": aggregates,
        "v340_over_v334": comparisons,
        "teacher_predictions_bit_exact": teacher_exact,
        "checks": checks,
        "passed": all(checks.values()),
        "authorizes_public_holdout_candidate_gate": all(checks.values()),
        "evidence_sha256": {
            "preregistration": sha256(args.preregistration),
            "recursive_ood_gate": sha256(args.recursive_ood_gate),
            "reward_checkpoint": sha256(args.reward_checkpoint),
        },
        "guards": {
            "public_train_episodes_only": True,
            "runtime_reads_reward_or_outcome": False,
            "official_batch16_outcomes_read": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "recursive_reanchor_count": len(changed), "v340_over_v334": comparisons,
        "reanchor": aggregates["v340"]["reanchor"], "checks": checks,
        "passed": report["passed"],
    }, indent=2), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
