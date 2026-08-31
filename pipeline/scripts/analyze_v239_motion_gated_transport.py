#!/usr/bin/env python3
"""Select a public-only motion gate for post-grasp transport retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wam_pipeline.arm_routed_autoregressive_runtime import Track2ArmRoutedAutoregressiveUNet


def corr(a: np.ndarray, b: np.ndarray) -> float | None:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def stats(x: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(x, dtype=np.float64)
    return {
        "count": int(x.size), "mean": float(x.mean()), "std": float(x.std()),
        "min": float(x.min()), "median": float(np.median(x)), "max": float(x.max()),
    }


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    files = sorted(args.audit_dir.glob("rollout_*.npz"))
    if len(files) != 50:
        raise ValueError(f"expected 50 captures, found {len(files)}")

    with np.load(args.library, allow_pickle=False) as values:
        paths = values["path"].astype(str)
        visuals = values["visual"].astype(np.float32)
        normalized = values["action"].astype(np.float32)
        mean = values["normalization_mean"].astype(np.float32)
        std = values["normalization_std"].astype(np.float32)
    steps = normalized.shape[1] // 14
    raw = normalized.reshape(-1, steps, 14) * std + mean
    lib_history, lib_future = raw[:, :-8], raw[:, -8:]
    lib_desc = descriptor(lib_history, lib_future)
    lib_path = path_length(lib_history, lib_future)
    lib_closed = (lib_future[..., 13] < 0.5).mean(-1)

    captures = []
    for path in files:
        with np.load(path, allow_pickle=False) as item:
            contexts = item["context_frames"]
            histories = item["history_actions"].astype(np.float32)
            futures = item["future_actions"].astype(np.float32)
            instructions = json.loads(str(item["instructions_json"]))
        for context, history, future, instruction in zip(contexts, histories, futures, instructions):
            route = Track2ArmRoutedAutoregressiveUNet.active_arm(history, future, str(instruction))
            post = route == "right" and history[-1, 13] < 0.5 and (future[:, 13] < 0.5).mean() >= 0.75
            if post:
                captures.append((visual(context[-1]), history, future))
    if len(captures) < 32:
        raise ValueError("insufficient post-grasp captures")

    reports = []
    for quantile in (0.50, 0.75, 0.90):
        threshold = float(np.quantile(lib_path, quantile))
        rows = np.flatnonzero((lib_closed >= 0.75) & (lib_path >= threshold))
        dmean, dstd = lib_desc[rows].mean(0), lib_desc[rows].std(0).clip(1e-4)
        lib_z = (lib_desc[rows] - dmean) / dstd
        distances, motions, alignments, selected_paths = [], [], [], []
        for query_visual, history, future in captures:
            qdesc = (descriptor(history[None], future[None])[0] - dmean) / dstd
            action_distance = ((lib_z - qdesc) ** 2).mean(1)
            visual_distance = ((visuals[rows] - query_visual) ** 2).mean(1)
            score = visual_distance / max(float(np.median(visual_distance)), 1e-9)
            score += 4.0 * action_distance / max(float(np.median(action_distance)), 1e-9)
            local = int(np.argmin(score))
            selected = int(rows[local])
            distances.append(float(action_distance[local]))
            motions.append(float(path_length(history[None], future[None])[0]))
            query_terminal = future[-1, 7:13] - history[-1, 7:13]
            library_terminal = lib_future[selected, -1, 7:13] - lib_history[selected, -1, 7:13]
            alignments.append(float(cosine(query_terminal[None], library_terminal[None])[0]))
            selected_paths.append(paths[selected])
        distances = np.asarray(distances)
        motions = np.asarray(motions)
        alignments = np.asarray(alignments)
        scale = float(np.median(distances))
        confidence = np.exp(-distances / (3.0 * scale))
        expert_similarity = -distances
        for motion_scale_name, motion_scale in (
            ("public_q50", float(np.quantile(lib_path[rows], 0.50))),
            ("public_q75", float(np.quantile(lib_path[rows], 0.75))),
            ("selected_min", threshold),
        ):
            gate = np.clip(motions / max(motion_scale, 1e-6), 0.0, 1.0)
            alpha = confidence * gate
            reports.append({
                "library_path_quantile": quantile,
                "library_rows": int(rows.size),
                "library_path_threshold": threshold,
                "motion_gate": motion_scale_name,
                "motion_scale": motion_scale,
                "distance": stats(distances),
                "query_motion": stats(motions),
                "directional_alignment": stats(alignments),
                "alpha": stats(alpha),
                "alpha_expert_similarity_correlation": corr(alpha, expert_similarity),
                "alpha_motion_correlation": corr(alpha, motions),
                "alpha_directional_alignment_correlation": corr(alpha, alignments),
                "expert_similarity_directional_alignment_correlation": corr(expert_similarity, alignments),
                "selected_unique": len(set(selected_paths)),
            })

    report = {
        "format": "strict-track2-v239-motion-gated-transport-analysis-v1",
        "source": "public v211 captures and declared public demonstration library",
        "post_grasp_queries": len(captures),
        "candidates": reports,
        "selection_rule": (
            "maximize the minimum of alpha correlations with expert similarity, motion, and directional alignment; "
            "all three correlations must be positive"
        ),
        "guards": {"policy_modified": False, "public_data_only": True,
                   "hidden_or_final_data": False, "real_submission": False},
    }
    valid = [r for r in reports if min(
        r["alpha_expert_similarity_correlation"], r["alpha_motion_correlation"],
        r["alpha_directional_alignment_correlation"]
    ) > 0]
    for item in valid:
        item["selection_score"] = min(
            item["alpha_expert_similarity_correlation"], item["alpha_motion_correlation"],
            item["alpha_directional_alignment_correlation"]
        )
    report["selected"] = max(valid, key=lambda x: x["selection_score"]) if valid else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
