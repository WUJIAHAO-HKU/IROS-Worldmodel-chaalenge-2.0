#!/usr/bin/env python3
"""Public-only reward sweep between the v209 parent and frozen v241 output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from wam_pipeline.arm_routed_autoregressive_runtime import Track2ArmRoutedAutoregressiveUNet
from wam_pipeline.rlinf_bridge.track2_client import Track2ServiceClient


SCALES = np.asarray([0.0, 0.5, 1.0, 2.0, 4.0, 8.0], dtype=np.float32)


def corr(a: np.ndarray, b: np.ndarray) -> float | None:
    a, b = np.asarray(a, dtype=np.float64).reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def stats(x: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return {"count": int(x.size), "mean": float(x.mean()), "std": float(x.std()),
            "min": float(x.min()), "median": float(np.median(x)), "max": float(x.max())}


def visual(frame: np.ndarray) -> np.ndarray:
    return (frame.astype(np.float32).reshape(16, 16, 16, 16, 3).mean((1, 3)) / 255).reshape(-1)


def descriptor(history: np.ndarray, future: np.ndarray) -> np.ndarray:
    return (future[..., 7:13] - history[..., -1:, 7:13]).reshape(future.shape[:-2] + (48,))


def path_length(history: np.ndarray, future: np.ndarray) -> np.ndarray:
    path = np.concatenate((history[..., -1:, 7:13], future[..., 7:13]), axis=-2)
    return np.linalg.norm(np.diff(path, axis=-2), axis=-1).sum(-1)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator = (a * b).sum(-1)
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-8)


def nchw(frames: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(frames).permute(0, 1, 4, 2, 3).reshape(-1, 3, 256, 256).float().div_(255)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--parent-url", required=True)
    parser.add_argument("--parent-version", required=True)
    parser.add_argument("--v241-url", required=True)
    parser.add_argument("--v241-version", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--reward-checkpoint", type=Path, required=True)
    parser.add_argument("--t5-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists() or args.details.exists():
        raise SystemExit("refusing to overwrite v242 outputs")

    with np.load(args.library, allow_pickle=False) as values:
        visuals = values["visual"].astype(np.float32)
        normalized = values["action"].astype(np.float32)
        norm_mean = values["normalization_mean"].astype(np.float32)
        norm_std = values["normalization_std"].astype(np.float32)
    steps = normalized.shape[1] // 14
    raw = normalized.reshape(-1, steps, 14) * norm_std + norm_mean
    lib_history, lib_future = raw[:, :-8], raw[:, -8:]
    lib_desc = descriptor(lib_history, lib_future)
    lib_path = path_length(lib_history, lib_future)
    closed = (lib_future[..., 13] < 0.5).mean(-1)
    threshold = float(np.quantile(lib_path, 0.90))
    rows = np.flatnonzero((closed >= 0.75) & (lib_path >= threshold))
    if rows.size != 188:
        raise ValueError(f"expected 188 frozen transport rows, got {rows.size}")
    dmean, dstd = lib_desc[rows].mean(0), lib_desc[rows].std(0).clip(1e-4)
    lib_z = (lib_desc[rows] - dmean) / dstd
    motion_scale = float(np.quantile(lib_path[rows], 0.75))

    records = []
    files = sorted(args.audit_dir.glob("rollout_*.npz"))
    if len(files) != 50:
        raise ValueError(f"expected 50 public captures, got {len(files)}")
    for file_index, path in enumerate(files):
        with np.load(path, allow_pickle=False) as item:
            contexts = item["context_frames"].copy()
            histories = item["history_actions"].astype(np.float32).copy()
            futures = item["future_actions"].astype(np.float32).copy()
            seeds = item["seeds"].astype(np.int64).copy()
            instructions = [str(x) for x in json.loads(str(item["instructions_json"]))]
        for action_index, (context, history, future, seed, instruction) in enumerate(
            zip(contexts, histories, futures, seeds, instructions)
        ):
            route = Track2ArmRoutedAutoregressiveUNet.active_arm(history, future, instruction)
            post = route == "right" and history[-1, 13] < 0.5 and (future[:, 13] < 0.5).mean() >= 0.75
            if post:
                records.append((file_index, action_index, context, history, future, int(seed), instruction))
    if len(records) != 101:
        raise ValueError(f"expected 101 frozen post-grasp queries, got {len(records)}")

    distances, motions, alignments, alphas, selected_rows = [], [], [], [], []
    for _, _, context, history, future, _, _ in records:
        query = (descriptor(history[None], future[None])[0] - dmean) / dstd
        action_distance = ((lib_z - query) ** 2).mean(1)
        visual_distance = ((visuals[rows] - visual(context[-1])) ** 2).mean(1)
        score = visual_distance / max(float(np.median(visual_distance)), 1e-9)
        score += 4.0 * action_distance / max(float(np.median(action_distance)), 1e-9)
        local = int(np.argmin(score))
        selected = int(rows[local])
        distance = float(action_distance[local])
        motion = float(path_length(history[None], future[None])[0])
        query_terminal = future[-1, 7:13] - history[-1, 7:13]
        lib_terminal = lib_future[selected, -1, 7:13] - lib_history[selected, -1, 7:13]
        alignment = float(cosine(query_terminal[None], lib_terminal[None])[0])
        confidence = float(np.exp(-distance / (10.0 * 26.628942489624023)))
        alignment_gate = float(np.clip((alignment - 0.5) / 0.5, 0.0, 1.0))
        motion_gate = float(np.clip(motion / motion_scale, 0.0, 1.0))
        distances.append(distance); motions.append(motion); alignments.append(alignment)
        alphas.append(confidence * alignment_gate * motion_gate); selected_rows.append(selected)
    distances, motions, alignments, alphas = map(
        np.asarray, (distances, motions, alignments, alphas)
    )

    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel
    reward_model = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        str(args.reward_checkpoint), config={"t5_model_name": str(args.t5_model)}
    ).eval().to(args.device)
    parent_client = Track2ServiceClient(args.parent_url, args.token, args.parent_version)
    v241_client = Track2ServiceClient(args.v241_url, args.token, args.v241_version)
    parent_client.assert_ready(); v241_client.assert_ready()
    rewards = np.empty((len(records), len(SCALES), 8), dtype=np.float32)
    with torch.inference_mode():
        for begin in range(0, len(records), args.batch_size):
            batch = records[begin:begin + args.batch_size]
            contexts = np.stack([x[2] for x in batch])
            histories = np.stack([x[3] for x in batch])
            futures = np.stack([x[4] for x in batch])
            seeds = np.asarray([x[5] for x in batch], dtype=np.int64)
            instructions = [x[6] for x in batch]
            parent = parent_client.predict_batch(contexts, histories, futures, seeds, instructions)
            modified = v241_client.predict_batch(contexts, histories, futures, seeds, instructions)
            candidates = []
            for scale in SCALES:
                candidate = np.clip(np.rint(
                    parent.astype(np.float32) + float(scale) *
                    (modified.astype(np.float32) - parent.astype(np.float32))
                ), 0, 255).astype(np.uint8)
                candidates.append(candidate)
            candidates = np.concatenate(candidates, axis=0)
            expanded = [text for _ in SCALES for text in instructions for _ in range(8)]
            score = reward_model.compute_reward(nchw(candidates).to(args.device), expanded)
            score = score.float().cpu().numpy().reshape(len(SCALES), len(batch), 8).transpose(1, 0, 2)
            rewards[begin:begin + len(batch)] = score
            print(json.dumps({"completed": begin + len(batch), "total": len(records)}), flush=True)

    terminal = rewards[..., -1]
    expert_similarity = -distances
    group_indices = []
    lookup = {(rec[0], rec[1]): i for i, rec in enumerate(records)}
    for file_index in range(len(files)):
        for group_begin in range(0, 8, 4):
            group = [lookup.get((file_index, j)) for j in range(group_begin, group_begin + 4)]
            if all(index is not None for index in group):
                group_indices.append(group)
    candidates_report = []
    for scale_index, scale in enumerate(SCALES):
        values = terminal[:, scale_index]
        group_stds = np.asarray([values[group].std() for group in group_indices])
        metrics = {
            "scale": float(scale), "terminal_reward": stats(values),
            "group_terminal_std": stats(group_stds),
            "reward_expert_similarity_correlation": corr(values, expert_similarity),
            "reward_motion_correlation": corr(values, motions),
            "reward_directional_alignment_correlation": corr(values, alignments),
            "reward_current_alpha_correlation": corr(values, alphas),
            "delta_from_parent_expert_similarity_correlation": corr(values - terminal[:, 0], expert_similarity),
            "delta_from_parent_motion_correlation": corr(values - terminal[:, 0], motions),
            "delta_from_parent_alignment_correlation": corr(values - terminal[:, 0], alignments),
        }
        correlations = [metrics["reward_expert_similarity_correlation"],
                        metrics["reward_motion_correlation"],
                        metrics["reward_directional_alignment_correlation"]]
        metrics["passes"] = bool(group_stds.mean() >= 1e-5 and all(
            value is not None and value >= 0.05 for value in correlations
        ))
        metrics["selection_score"] = float(min(correlations)) if all(
            value is not None for value in correlations
        ) else None
        candidates_report.append(metrics)
    valid = [x for x in candidates_report if x["passes"]]
    selected = max(valid, key=lambda x: x["selection_score"]) if valid else None
    report = {
        "format": "strict-track2-v242-public-reward-blend-scale-sweep-v1",
        "post_grasp_queries": len(records), "complete_post_grasp_groups": len(group_indices),
        "frozen_scales": SCALES.tolist(), "current_alpha": stats(alphas),
        "query_motion": stats(motions), "expert_similarity": stats(expert_similarity),
        "directional_alignment": stats(alignments), "selected_unique_rows": len(set(selected_rows)),
        "selection_rule": "group std >=1e-5 and reward correlations with expert similarity, motion, alignment all >=0.05; maximize minimum correlation",
        "candidates": candidates_report, "selected": selected,
        "guards": {"public_capture_only": True, "policy_modified": False,
                   "hidden_or_final_data": False, "real_submission": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.details, scales=SCALES, rewards=rewards, distances=distances,
                        motions=motions, alignments=alignments, alphas=alphas,
                        selected_rows=np.asarray(selected_rows, dtype=np.int64))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
