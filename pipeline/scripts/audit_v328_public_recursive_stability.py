#!/usr/bin/env python3
"""Replay held-out public right-arm demonstrations recursively through world models.

The bridge feeds prediction frames 3..7 back as the next five-frame context.
This audit reproduces that update with fixed public expert actions at starts
0,8,16,... and compares v328 with frozen v326/v317.  It is an offline public
gate only and never reads contest evaluation outcomes.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from audit_v310_full_mirror_causal_gate import RIGHT_EPISODES, score_terminal, sha256
from wam_pipeline.v317_batched_sparse_failure_terminal_runtime import (
    Track2V317BatchedSparseFailureTerminal,
)
from wam_pipeline.v326_blended_phase_terminal_runtime import (
    Track2V326BlendedPhaseTerminal,
)
from wam_pipeline.v328_coherent_phase_trajectory_runtime import (
    Track2V328CoherentPhaseTrajectory,
)


def seed(path: Path) -> int:
    return int.from_bytes(hashlib.sha256(path.name.encode()).digest()[:8], "little") % (2**31)


def rgb_mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.abs(prediction.astype(np.int16) - target.astype(np.int16)).mean())


def temporal_delta_error(prediction: np.ndarray, target: np.ndarray) -> float:
    predicted_delta = np.diff(prediction.astype(np.float32), axis=0)
    target_delta = np.diff(target.astype(np.float32), axis=0)
    return float(np.abs(predicted_delta - target_delta).mean())


def summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0, "mean": None, "median": None, "p90": None}
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
    }


def instantiate(name: str, args):
    if name == "v317":
        return Track2V317BatchedSparseFailureTerminal(
            args.checkpoint_dir, args.library_index, args.device, args.action_gate
        )
    if name == "v326":
        return Track2V326BlendedPhaseTerminal(
            args.checkpoint_dir,
            args.library_index,
            args.device,
            args.action_gate,
            args.phase_gate,
        )
    if name == "v328":
        return Track2V328CoherentPhaseTrajectory(
            args.checkpoint_dir,
            args.library_index,
            args.device,
            args.action_gate,
            args.phase_gate,
        )
    raise ValueError(name)


def run_model(name: str, args, instructions: dict[str, str]) -> list[dict]:
    runtime = instantiate(name, args)
    rows = []
    reward_frames = []
    reward_prompts = []
    for split_name, episodes in RIGHT_EPISODES.items():
        for episode in episodes:
            paths = sorted(
                args.windows.glob(f"episode{episode}_*.npz"),
                key=lambda path: int(path.stem.split("_")[1]),
            )
            paths = [path for path in paths if int(path.stem.split("_")[1]) % 8 == 0]
            if not paths or int(paths[0].stem.split("_")[1]) != 0:
                raise RuntimeError(f"episode {episode} has no start-zero replay")
            recursive_context = None
            prompt = str(instructions[str(episode)])
            for chunk_index, path in enumerate(paths):
                start = int(path.stem.split("_")[1])
                if start != chunk_index * 8:
                    raise RuntimeError(f"episode {episode} replay is not contiguous at {start}")
                with np.load(path, allow_pickle=False) as values:
                    real_context = values["context_frames"].astype(np.uint8)
                    history = values["history_actions"].astype(np.float32)
                    future = values["future_actions"].astype(np.float32)
                    target = values["target_frames"].astype(np.uint8)
                if recursive_context is None:
                    recursive_context = real_context.copy()
                teacher_phase_override = False
                if name == "v328":
                    probability = runtime._probability(history, future)
                    signature = runtime._signature(history, future, probability)
                    teacher_phase_override = runtime._coherent_phase_override(
                        real_context, history, future, probability, signature
                    )
                contexts = np.stack((real_context, recursive_context))
                predictions = runtime.predict_batch(
                    contexts,
                    np.repeat(history[None], 2, axis=0),
                    np.repeat(future[None], 2, axis=0),
                    np.repeat(seed(path), 2),
                    [prompt, prompt],
                )
                teacher, recursive = predictions
                next_target = target[-5:]
                row = {
                    "key": f"{split_name}/episode{episode}/{start:05d}",
                    "split": split_name,
                    "episode": episode,
                    "start": start,
                    "chunk_index": chunk_index,
                    "teacher_phase_override": teacher_phase_override,
                    "teacher_next_context_rgb_mae": rgb_mae(teacher[-5:], next_target),
                    "recursive_next_context_rgb_mae": rgb_mae(recursive[-5:], next_target),
                    "teacher_next_context_temporal_delta_error": temporal_delta_error(
                        teacher[-5:], next_target
                    ),
                    "recursive_next_context_temporal_delta_error": temporal_delta_error(
                        recursive[-5:], next_target
                    ),
                    "recursive_vs_teacher_next_context_rgb_mae": rgb_mae(
                        recursive[-5:], teacher[-5:]
                    ),
                }
                row["reward_frame_offset"] = len(reward_frames)
                reward_frames.extend((real_context[-1], target[-1], teacher[-1], recursive[-1]))
                reward_prompts.extend((prompt, prompt, prompt, prompt))
                rows.append(row)
                recursive_context = recursive[-5:].copy()

    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel

    reward = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        str(args.reward_checkpoint), config={"t5_model_name": str(args.t5_model)}
    ).to(args.device).eval().requires_grad_(False)
    scores = score_terminal(
        reward,
        np.stack(reward_frames),
        reward_prompts,
        torch.device(args.device),
        args.reward_batch_size,
    )
    for row in rows:
        offset = row.pop("reward_frame_offset")
        values = scores[offset : offset + 4]
        row.update({
            "context_terminal_reward": float(values[0]),
            "gt_terminal_reward": float(values[1]),
            "teacher_terminal_reward": float(values[2]),
            "recursive_terminal_reward": float(values[3]),
            "teacher_reward_absolute_error": float(abs(values[2] - values[1])),
            "recursive_reward_absolute_error": float(abs(values[3] - values[1])),
        })
    del reward, runtime
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    print(f"RECURSIVE_MODEL_SCORED {name} {len(rows)}", flush=True)
    return rows


def aggregate(rows: list[dict], phase_keys: set[str]) -> dict:
    result = {}
    for split_name in RIGHT_EPISODES:
        split_rows = [row for row in rows if row["split"] == split_name]
        groups = {
            "all": split_rows,
            "phase_qualified": [row for row in split_rows if row["key"] in phase_keys],
        }
        result[split_name] = {}
        for group_name, selected in groups.items():
            metrics = {
                key: summary([float(row[key]) for row in selected])
                for key in (
                    "teacher_next_context_rgb_mae",
                    "recursive_next_context_rgb_mae",
                    "teacher_next_context_temporal_delta_error",
                    "recursive_next_context_temporal_delta_error",
                    "recursive_vs_teacher_next_context_rgb_mae",
                    "teacher_reward_absolute_error",
                    "recursive_reward_absolute_error",
                    "teacher_terminal_reward",
                    "recursive_terminal_reward",
                )
            }
            metrics["recursive_reward_hit_rate_at_0p9"] = (
                float(np.mean([row["recursive_terminal_reward"] >= 0.90 for row in selected]))
                if selected else None
            )
            result[split_name][group_name] = metrics
    return result


def mean(aggregates: dict, model: str, split: str, group: str, metric: str) -> float:
    value = aggregates[model][split][group][metric]["mean"]
    if value is None:
        raise RuntimeError(f"empty aggregate: {model}/{split}/{group}/{metric}")
    return float(value)


def ratio(value: float, baseline: float) -> float:
    return float(value / max(baseline, 1e-9))


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "checkpoint-dir", "library-index", "action-gate", "phase-gate", "windows",
        "instruction-map", "reward-checkpoint", "t5-model", "contract-report",
        "causal-report", "output",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reward-batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    contract = json.loads(args.contract_report.read_text())
    causal = json.loads(args.causal_report.read_text())
    if contract.get("passed") is not True or causal.get("passed") is not True:
        raise RuntimeError("v328 prerequisite gate failed")
    instruction_payload = json.loads(args.instruction_map.read_text())
    if instruction_payload.get("hidden_or_final_evaluation_data") is not False:
        raise RuntimeError("instruction map data boundary is invalid")
    instructions = instruction_payload["episode_to_instruction"]

    model_rows = {name: run_model(name, args, instructions) for name in ("v317", "v326", "v328")}
    keys = [[row["key"] for row in model_rows[name]] for name in model_rows]
    if not all(value == keys[0] for value in keys[1:]):
        raise RuntimeError("model replay rows are misaligned")
    phase_keys = {
        row["key"] for row in model_rows["v328"] if row["teacher_phase_override"]
    }
    aggregates = {
        name: aggregate(rows, phase_keys) for name, rows in model_rows.items()
    }
    comparisons = {}
    for split_name in RIGHT_EPISODES:
        comparisons[split_name] = {}
        for group_name in ("all", "phase_qualified"):
            comparisons[split_name][group_name] = {
                metric: ratio(
                    mean(aggregates, "v328", split_name, group_name, metric),
                    mean(aggregates, "v326", split_name, group_name, metric),
                )
                for metric in (
                    "recursive_next_context_rgb_mae",
                    "recursive_next_context_temporal_delta_error",
                    "recursive_reward_absolute_error",
                )
            }
    phase_counts = {
        split_name: aggregates["v328"][split_name]["phase_qualified"]
        ["recursive_next_context_rgb_mae"]["count"]
        for split_name in RIGHT_EPISODES
    }
    checks = {
        "contract_gate": contract.get("passed") is True,
        "causal_gate": causal.get("passed") is True,
        "validation_phase_qualified_windows_ge_2": phase_counts["validation"] >= 2,
        "local_test_phase_qualified_windows_ge_2": phase_counts["local_test"] >= 2,
        "validation_phase_rgb_ratio_le_0p90": comparisons["validation"]["phase_qualified"]
        ["recursive_next_context_rgb_mae"] <= 0.90,
        "local_test_phase_rgb_ratio_le_0p90": comparisons["local_test"]["phase_qualified"]
        ["recursive_next_context_rgb_mae"] <= 0.90,
        "validation_phase_temporal_ratio_le_0p90": comparisons["validation"]["phase_qualified"]
        ["recursive_next_context_temporal_delta_error"] <= 0.90,
        "local_test_phase_temporal_ratio_le_0p90": comparisons["local_test"]["phase_qualified"]
        ["recursive_next_context_temporal_delta_error"] <= 0.90,
        "validation_phase_reward_error_nonregression": comparisons["validation"]["phase_qualified"]
        ["recursive_reward_absolute_error"] <= 1.0,
        "local_test_phase_reward_error_nonregression": comparisons["local_test"]["phase_qualified"]
        ["recursive_reward_absolute_error"] <= 1.0,
        "validation_all_rgb_ratio_le_1p02": comparisons["validation"]["all"]
        ["recursive_next_context_rgb_mae"] <= 1.02,
        "local_test_all_rgb_ratio_le_1p02": comparisons["local_test"]["all"]
        ["recursive_next_context_rgb_mae"] <= 1.02,
        "validation_all_reward_error_ratio_le_1p02": comparisons["validation"]["all"]
        ["recursive_reward_absolute_error"] <= 1.02,
        "local_test_all_reward_error_ratio_le_1p02": comparisons["local_test"]["all"]
        ["recursive_reward_absolute_error"] <= 1.02,
    }
    report = {
        "format": "strict-track2-v328-public-recursive-stability-gate-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": "track2-v328-coherent-phase-trajectory-v326-v317",
        "replay_rule": {
            "episodes": RIGHT_EPISODES,
            "starts": "0,8,16,... through the final complete public window",
            "actions": "frozen public expert future_actions",
            "recursive_update": "next context equals prediction[-5:] (frames 3..7)",
            "teacher_phase_subset": "frozen v328 action/phase/failure gates on public ground-truth context",
        },
        "phase_qualified_counts": phase_counts,
        "aggregates": aggregates,
        "v328_over_v326_comparisons": comparisons,
        "checks": checks,
        "passed": all(checks.values()),
        "authorization": {
            "expensive_rl_allowed": all(checks.values()),
            "all_checks_required": True,
            "waivers_or_excluded_failed_checks": False,
        },
        "evidence_sha256": {
            "contract_report": sha256(args.contract_report),
            "causal_report": sha256(args.causal_report),
            "action_gate": sha256(args.action_gate),
            "phase_gate": sha256(args.phase_gate),
            "instruction_map": sha256(args.instruction_map),
            "reward_checkpoint": sha256(args.reward_checkpoint),
        },
        "guards": {
            "declared_public_world_model_data_only": True,
            "runtime_reads_reward_or_outcome": False,
            "public_evaluation_outcomes_read": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
        "rows": model_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "passed": report["passed"],
        "phase_qualified_counts": phase_counts,
        "v328_over_v326_comparisons": comparisons,
        "checks": checks,
    }, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
