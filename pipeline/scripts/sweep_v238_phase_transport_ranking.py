#!/usr/bin/env python3
"""Rank post-grasp right-arm transport retrievals on public captures."""

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
    value = frame.astype(np.float32).reshape(16, 16, 16, 16, 3).mean((1, 3)) / 255.0
    return value.reshape(-1)


def stats(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not values.size:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(values.size), "mean": float(values.mean()),
        "std": float(values.std()), "min": float(values.min()), "max": float(values.max()),
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
    return torch.from_numpy(frames).permute(0, 1, 4, 2, 3).reshape(-1, 3, 256, 256).float().div_(255.0)


def blend(parent: np.ndarray, target: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha = np.broadcast_to(np.asarray(alpha, dtype=np.float32), parent.shape[:2])
    alpha = alpha.reshape(parent.shape[:2] + (1, 1, 1))
    return np.clip(
        np.rint((1.0 - alpha) * parent.astype(np.float32) + alpha * target.astype(np.float32)), 0, 255
    ).astype(np.uint8)


def transport_descriptor(history: np.ndarray, future: np.ndarray) -> np.ndarray:
    relative = future[..., 7:13] - history[..., -1:, 7:13]
    return relative.reshape(relative.shape[:-2] + (48,))


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

    library_steps = library_action.shape[1] // 14
    library_raw = library_action.reshape(-1, library_steps, 14) * action_std + action_mean
    library_history, library_future = library_raw[:, :-8], library_raw[:, -8:]
    library_closed = (library_future[..., 13] < 0.5).mean(1)
    library_path = np.linalg.norm(
        np.diff(np.concatenate((library_history[:, -1:, 7:13], library_future[..., 7:13]), axis=1), axis=1), axis=2
    ).sum(1)
    transport_rows = np.flatnonzero(
        (library_closed >= 0.75) & (library_path >= np.median(library_path))
    )
    if transport_rows.size < 128:
        raise ValueError("insufficient public closed-transport windows")
    library_transport = transport_descriptor(library_history, library_future)
    transport_mean = library_transport[transport_rows].mean(0)
    transport_std = library_transport[transport_rows].std(0).clip(1e-4)
    library_transport_z = (library_transport[transport_rows] - transport_mean) / transport_std

    original_selections, transport_selections = [], []
    transport_distances, route_rows, post_rows = [], [], []
    for path in files:
        with np.load(path, allow_pickle=False) as item:
            contexts = item["context_frames"]
            histories = item["history_actions"].astype(np.float32)
            actions = item["future_actions"].astype(np.float32)
            instructions = json.loads(str(item["instructions_json"]))
        file_original, file_transport, file_distance = [], [], []
        file_routes, file_post = [], []
        for context, history, future, instruction in zip(contexts, histories, actions, instructions):
            route = Track2ArmRoutedAutoregressiveUNet.active_arm(history, future, str(instruction))
            query_visual = visual_descriptor(context[-1])
            query_actions = ((np.concatenate((history, future), axis=0) - action_mean) / action_std).reshape(-1)
            visual_distance = ((library_visual - query_visual) ** 2).mean(1)
            action_distance = ((library_action - query_actions) ** 2).mean(1)
            original_score = visual_distance / max(float(np.median(visual_distance)), 1e-9)
            original_score += 2.5 * action_distance / max(float(np.median(action_distance)), 1e-9)
            original = int(np.argmin(original_score))

            query_transport = (transport_descriptor(history[None], future[None])[0] - transport_mean) / transport_std
            phase_distance = ((library_transport_z - query_transport) ** 2).mean(1)
            phase_visual = visual_distance[transport_rows]
            phase_score = phase_visual / max(float(np.median(phase_visual)), 1e-9)
            phase_score += 4.0 * phase_distance / max(float(np.median(phase_distance)), 1e-9)
            local = int(np.argmin(phase_score))
            transport = int(transport_rows[local])
            post = bool(
                route == "right" and history[-1, 13] < 0.5
                and (future[:, 13] < 0.5).mean() >= 0.75
            )
            file_original.append(original)
            file_transport.append(transport)
            file_distance.append(float(phase_distance[local]))
            file_routes.append(route)
            file_post.append(post)
        original_selections.append(file_original)
        transport_selections.append(file_transport)
        transport_distances.append(file_distance)
        route_rows.append(file_routes)
        post_rows.append(file_post)

    original_selections = np.asarray(original_selections, dtype=np.int64)
    transport_selections = np.asarray(transport_selections, dtype=np.int64)
    transport_distances = np.asarray(transport_distances, dtype=np.float32)
    route_rows = np.asarray(route_rows)
    post_rows = np.asarray(post_rows, dtype=bool)
    right_mask = route_rows == "right"
    distance_scale = float(np.median(transport_distances[post_rows]))
    if distance_scale <= 0:
        raise ValueError("invalid post-grasp transport distance scale")

    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel
    reward_model = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        str(args.reward_checkpoint), config={"t5_model_name": str(args.t5_model)}
    ).eval().to(args.device)
    names = (
        "parent", "v236_causal070_raw", "phase070_raw", "phase070_delta",
        "phase_conf070_raw", "phase_conf100_raw", "phase_ramp100_raw", "phase_conf100_delta",
    )
    rewards = {name: [] for name in names}
    actions_all = []
    target_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def target_for(index: int) -> tuple[np.ndarray, np.ndarray]:
        cached = target_cache.get(index)
        if cached is None:
            with np.load(library_paths[index], allow_pickle=False) as item:
                cached = (item["context_frames"][-1].copy(), item["target_frames"].copy())
            target_cache[index] = cached
        return cached

    with torch.inference_mode():
        for file_index, path in enumerate(files):
            with np.load(path, allow_pickle=False) as item:
                contexts = item["context_frames"].copy()
                actions = item["future_actions"].astype(np.float32).copy()
                parent = item["predicted_frames"].copy()
                instructions = [str(x) for x in json.loads(str(item["instructions_json"]))]
            original_raw, phase_raw, phase_delta = [], [], []
            for row, (original, transport) in enumerate(zip(original_selections[file_index], transport_selections[file_index])):
                _, original_target = target_for(int(original))
                phase_context, phase_target = target_for(int(transport))
                original_raw.append(original_target)
                phase_raw.append(phase_target)
                residual = phase_target.astype(np.int16) - phase_context.astype(np.int16)[None]
                phase_delta.append(
                    np.clip(contexts[row, -1].astype(np.int16)[None] + residual, 0, 255).astype(np.uint8)
                )
            original_raw = np.stack(original_raw)
            phase_raw = np.stack(phase_raw)
            phase_delta = np.stack(phase_delta)
            route = right_mask[file_index]
            post = post_rows[file_index]
            closed = actions[..., 13] < 0.5
            causal = np.where(route[:, None], closed, False).astype(np.float32)
            post_causal = np.where(post[:, None], closed, False).astype(np.float32)
            confidence = np.exp(-transport_distances[file_index] / (3.0 * distance_scale)).astype(np.float32)
            base = blend(parent, original_raw, 0.70 * causal)
            time_ramp = np.linspace(0.25, 1.0, 8, dtype=np.float32)[None]

            def phase_candidate(target: np.ndarray, alpha: np.ndarray) -> np.ndarray:
                result = base.copy()
                if post.any():
                    replacement = blend(parent, target, alpha)
                    result[post] = replacement[post]
                return result

            candidates = {
                "parent": parent,
                "v236_causal070_raw": base,
                "phase070_raw": phase_candidate(phase_raw, 0.70 * post_causal),
                "phase070_delta": phase_candidate(phase_delta, 0.70 * post_causal),
                "phase_conf070_raw": phase_candidate(phase_raw, 0.70 * post_causal * confidence[:, None]),
                "phase_conf100_raw": phase_candidate(phase_raw, post_causal * confidence[:, None]),
                "phase_ramp100_raw": phase_candidate(phase_raw, post_causal * confidence[:, None] * time_ramp),
                "phase_conf100_delta": phase_candidate(phase_delta, post_causal * confidence[:, None]),
            }
            expanded = [text for text in instructions for _ in range(actions.shape[1])]
            for name, frames in candidates.items():
                score = reward_model.compute_reward(to_nchw(frames).to(args.device), expanded)
                rewards[name].append(score.float().cpu().numpy().reshape(actions.shape[:2]))
            actions_all.append(actions)
            print(json.dumps({"completed": file_index + 1, "total": len(files)}), flush=True)

    actions_all = np.stack(actions_all)
    reward_arrays = {name: np.stack(rows) for name, rows in rewards.items()}
    right_motion = np.linalg.norm(actions_all[..., -1, 7:13] - actions_all[..., 0, 7:13], axis=-1)
    expert_similarity = -transport_distances
    reports = []
    for name in names:
        terminal = reward_arrays[name][..., -1]
        group_stds = []
        for chunk in range(terminal.shape[0]):
            for begin in range(0, terminal.shape[1], args.group_size):
                stop = begin + args.group_size
                if stop <= terminal.shape[1] and post_rows[chunk, begin:stop].all():
                    group_stds.append(float(terminal[chunk, begin:stop].std()))
        reports.append({
            "name": name,
            "right_terminal_reward": stats(terminal[right_mask]),
            "post_grasp_terminal_reward": stats(terminal[post_rows]),
            "post_grasp_group_terminal_std": stats(np.asarray(group_stds)),
            "post_grasp_expert_similarity_correlation": correlation(expert_similarity[post_rows], terminal[post_rows]),
            "post_grasp_motion_correlation": correlation(right_motion[post_rows], terminal[post_rows]),
        })

    report = {
        "format": "strict-track2-v238-phase-transport-ranking-v1",
        "source": "public v211 policy captures and declared public demonstrations",
        "capture_files": len(files), "library_rows": len(library_paths),
        "transport_library_rows": int(transport_rows.size),
        "right_query_count": int(right_mask.sum()),
        "post_grasp_query_count": int(post_rows.sum()),
        "post_grasp_transport_distance": stats(transport_distances[post_rows]),
        "distance_scale": distance_scale, "candidates": reports,
        "selection_rule": (
            "maximize post-grasp expert-similarity correlation and within-group variance; "
            "require nonnegative post-grasp motion correlation"
        ),
        "guards": {"policy_modified": False, "world_model_service_modified": False,
                   "public_data_only": True, "hidden_or_final_data": False, "real_submission": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
