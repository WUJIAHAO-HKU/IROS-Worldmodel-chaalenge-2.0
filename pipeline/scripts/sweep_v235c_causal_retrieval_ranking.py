#!/usr/bin/env python3
"""Sweep causal/confidence-gated public retrieval using captured policy actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from wam_pipeline.arm_routed_autoregressive_runtime import (
    Track2ArmRoutedAutoregressiveUNet,
)


def visual_descriptor(frame: np.ndarray) -> np.ndarray:
    value = (
        frame.astype(np.float32)
        .reshape(16, 16, 16, 16, 3)
        .mean((1, 3))
        / 255.0
    )
    return value.reshape(-1)


def stats(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not values.size:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    valid = np.isfinite(first) & np.isfinite(second)
    first, second = first[valid], second[valid]
    if first.size < 2 or first.std() == 0.0 or second.std() == 0.0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def to_nchw(frames: np.ndarray) -> torch.Tensor:
    return (
        torch.from_numpy(frames)
        .permute(0, 1, 4, 2, 3)
        .reshape(-1, 3, frames.shape[2], frames.shape[3])
        .float()
        .div_(255.0)
    )


def blend(parent: np.ndarray, target: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha = np.asarray(alpha, dtype=np.float32)
    try:
        alpha = np.broadcast_to(alpha, parent.shape[:2])
    except ValueError as exc:
        raise ValueError(
            f"alpha must have shape {parent.shape[:2]}, got {alpha.shape}"
        ) from exc
    alpha = alpha.reshape(parent.shape[:2] + (1, 1, 1))
    return np.clip(
        np.rint(
            (1.0 - alpha) * parent.astype(np.float32)
            + alpha * target.astype(np.float32)
        ),
        0,
        255,
    ).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--reward-checkpoint", type=Path, required=True)
    parser.add_argument("--t5-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--group-size", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    files = sorted(args.audit_dir.glob("rollout_*.npz"))
    if len(files) != 50:
        raise ValueError(f"expected 50 capture files, found {len(files)}")
    with np.load(args.library, allow_pickle=False) as values:
        library_paths = values["path"].astype(str)
        library_visual = values["visual"].astype(np.float32)
        library_action = values["action"].astype(np.float32)
        action_mean = values["normalization_mean"].astype(np.float32)
        action_std = values["normalization_std"].astype(np.float32)

    selections = []
    selected_action_distances = []
    route_rows = []
    for path in files:
        with np.load(path, allow_pickle=False) as item:
            contexts = item["context_frames"]
            histories = item["history_actions"].astype(np.float32)
            actions = item["future_actions"].astype(np.float32)
            instructions = json.loads(str(item["instructions_json"]))
        file_selection, file_distance, file_routes = [], [], []
        for context, history, future, instruction in zip(
            contexts, histories, actions, instructions
        ):
            route = Track2ArmRoutedAutoregressiveUNet.active_arm(
                history, future, str(instruction)
            )
            query_visual = visual_descriptor(context[-1])
            query_actions = (
                (np.concatenate((history, future), axis=0) - action_mean)
                / action_std
            ).reshape(-1)
            visual_distance = ((library_visual - query_visual) ** 2).mean(1)
            action_distance = ((library_action - query_actions) ** 2).mean(1)
            score = visual_distance / max(float(np.median(visual_distance)), 1e-9)
            score += 2.5 * action_distance / max(
                float(np.median(action_distance)), 1e-9
            )
            selected = int(np.argmin(score))
            file_selection.append(selected)
            file_distance.append(float(action_distance[selected]))
            file_routes.append(route)
        selections.append(file_selection)
        selected_action_distances.append(file_distance)
        route_rows.append(file_routes)
    selections = np.asarray(selections, dtype=np.int64)
    selected_action_distances = np.asarray(selected_action_distances)
    route_rows = np.asarray(route_rows)
    right_mask = route_rows == "right"
    distance_scale = float(np.median(selected_action_distances[right_mask]))
    if distance_scale <= 0:
        raise ValueError("invalid selected action-distance scale")

    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel

    reward_model = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        str(args.reward_checkpoint),
        config={"t5_model_name": str(args.t5_model)},
    ).eval().to(args.device)

    names = (
        "parent",
        "fixed070_raw",
        "causal070_raw",
        "causal_soft070_raw",
        "causal_hard070_raw",
        "fixed070_delta",
        "causal070_delta",
        "causal_soft070_delta",
        "causal_soft100_delta",
        "causal_hard070_delta",
    )
    rewards = {name: [] for name in names}
    actions_all, instructions_all = [], []
    target_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def target_for(index: int) -> tuple[np.ndarray, np.ndarray]:
        cached = target_cache.get(index)
        if cached is None:
            with np.load(library_paths[index], allow_pickle=False) as item:
                cached = (
                    item["context_frames"][-1].copy(),
                    item["target_frames"].copy(),
                )
            target_cache[index] = cached
        return cached

    with torch.inference_mode():
        for file_index, path in enumerate(files):
            with np.load(path, allow_pickle=False) as item:
                contexts = item["context_frames"].copy()
                histories = item["history_actions"].astype(np.float32).copy()
                actions = item["future_actions"].astype(np.float32).copy()
                parent = item["predicted_frames"].copy()
                instructions = [str(x) for x in json.loads(str(item["instructions_json"]))]
            target_raw, target_delta = [], []
            for row, selected in enumerate(selections[file_index]):
                library_context, library_target = target_for(int(selected))
                target_raw.append(library_target)
                residual = library_target.astype(np.int16) - library_context.astype(
                    np.int16
                )[None]
                target_delta.append(
                    np.clip(
                        contexts[row, -1].astype(np.int16)[None] + residual,
                        0,
                        255,
                    ).astype(np.uint8)
                )
            target_raw = np.stack(target_raw)
            target_delta = np.stack(target_delta)

            route = right_mask[file_index]
            closed = actions[..., 13] < 0.5
            causal = np.where(route[:, None], closed, False).astype(np.float32)
            fixed = np.where(route[:, None], 1.0, 0.0).astype(np.float32)
            confidence = np.exp(
                -selected_action_distances[file_index] / (3.0 * distance_scale)
            ).astype(np.float32)
            soft = causal * confidence[:, None]
            hard = causal * (
                selected_action_distances[file_index] <= distance_scale
            )[:, None]

            candidates = {
                "parent": parent,
                "fixed070_raw": blend(parent, target_raw, 0.70 * fixed),
                "causal070_raw": blend(parent, target_raw, 0.70 * causal),
                "causal_soft070_raw": blend(parent, target_raw, 0.70 * soft),
                "causal_hard070_raw": blend(parent, target_raw, 0.70 * hard),
                "fixed070_delta": blend(parent, target_delta, 0.70 * fixed),
                "causal070_delta": blend(parent, target_delta, 0.70 * causal),
                "causal_soft070_delta": blend(parent, target_delta, 0.70 * soft),
                "causal_soft100_delta": blend(parent, target_delta, 1.00 * soft),
                "causal_hard070_delta": blend(parent, target_delta, 0.70 * hard),
            }
            expanded = [text for text in instructions for _ in range(actions.shape[1])]
            for name, frames in candidates.items():
                score = reward_model.compute_reward(
                    to_nchw(frames).to(args.device), expanded
                ).float().cpu().numpy().reshape(actions.shape[:2])
                rewards[name].append(score)
            actions_all.append(actions)
            instructions_all.append(instructions)
            print(json.dumps({"completed": file_index + 1, "total": len(files)}), flush=True)

    actions_all = np.stack(actions_all)
    instructions_all = np.asarray(instructions_all, dtype=object)
    reward_arrays = {name: np.stack(rows) for name, rows in rewards.items()}
    right_close = (actions_all[..., 13] < 0.5).mean(axis=-1)
    right_motion = np.linalg.norm(
        np.diff(actions_all[..., 7:13], axis=2), axis=-1
    ).mean(axis=-1)
    expert_similarity = -selected_action_distances

    candidate_reports = []
    for name in names:
        value = reward_arrays[name]
        terminal = value[..., -1]
        group_stds = []
        for chunk in range(value.shape[0]):
            for begin in range(0, value.shape[1], args.group_size):
                stop = begin + args.group_size
                if stop <= value.shape[1] and right_mask[
                    chunk, begin:stop
                ].all():
                    group_stds.append(float(terminal[chunk, begin:stop].std()))
        candidate_reports.append(
            {
                "name": name,
                "right_frame_reward": stats(value[right_mask]),
                "right_terminal_reward": stats(terminal[right_mask]),
                "right_group_terminal_std": stats(np.asarray(group_stds)),
                "right_expert_similarity_terminal_correlation": correlation(
                    expert_similarity[right_mask], terminal[right_mask]
                ),
                "right_close_terminal_correlation": correlation(
                    right_close[right_mask], terminal[right_mask]
                ),
                "right_motion_terminal_correlation": correlation(
                    right_motion[right_mask], terminal[right_mask]
                ),
            }
        )

    report = {
        "format": "strict-track2-v235-causal-retrieval-ranking-sweep-v1",
        "source": "public v211 policy captures and declared public demonstrations",
        "capture_files": len(files),
        "right_query_count": int(right_mask.sum()),
        "selected_action_distance": stats(selected_action_distances[right_mask]),
        "distance_scale": distance_scale,
        "candidates": candidate_reports,
        "selection_rule": (
            "maximize right expert-similarity/reward correlation; require nonzero "
            "within-group reward variance and no worse motion correlation than parent"
        ),
        "guards": {
            "policy_modified": False,
            "world_model_service_modified": False,
            "public_data_only": True,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
