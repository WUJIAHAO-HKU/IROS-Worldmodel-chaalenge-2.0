#!/usr/bin/env python3
"""Select an expert-aligned motion gate for public post-grasp retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analyze_v239_motion_gated_transport as v239
from wam_pipeline.arm_routed_autoregressive_runtime import Track2ArmRoutedAutoregressiveUNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    files = sorted(args.audit_dir.glob("rollout_*.npz"))
    with np.load(args.library, allow_pickle=False) as values:
        paths = values["path"].astype(str)
        visuals = values["visual"].astype(np.float32)
        normalized = values["action"].astype(np.float32)
        mean = values["normalization_mean"].astype(np.float32)
        std = values["normalization_std"].astype(np.float32)
    steps = normalized.shape[1] // 14
    raw = normalized.reshape(-1, steps, 14) * std + mean
    lib_history, lib_future = raw[:, :-8], raw[:, -8:]
    lib_desc = v239.descriptor(lib_history, lib_future)
    lib_path = v239.path_length(lib_history, lib_future)
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
            if route == "right" and history[-1, 13] < 0.5 and (future[:, 13] < 0.5).mean() >= 0.75:
                captures.append((v239.visual(context[-1]), history, future))

    reports = []
    for quantile in (0.50, 0.75, 0.90):
        threshold = float(np.quantile(lib_path, quantile))
        rows = np.flatnonzero((lib_closed >= 0.75) & (lib_path >= threshold))
        dmean, dstd = lib_desc[rows].mean(0), lib_desc[rows].std(0).clip(1e-4)
        lib_z = (lib_desc[rows] - dmean) / dstd
        distances, motions, alignments = [], [], []
        for query_visual, history, future in captures:
            qdesc = (v239.descriptor(history[None], future[None])[0] - dmean) / dstd
            action_distance = ((lib_z - qdesc) ** 2).mean(1)
            visual_distance = ((visuals[rows] - query_visual) ** 2).mean(1)
            score = visual_distance / max(float(np.median(visual_distance)), 1e-9)
            score += 4.0 * action_distance / max(float(np.median(action_distance)), 1e-9)
            local = int(np.argmin(score))
            selected = int(rows[local])
            distances.append(float(action_distance[local]))
            motions.append(float(v239.path_length(history[None], future[None])[0]))
            query_terminal = future[-1, 7:13] - history[-1, 7:13]
            library_terminal = lib_future[selected, -1, 7:13] - lib_history[selected, -1, 7:13]
            alignments.append(float(v239.cosine(query_terminal[None], library_terminal[None])[0]))
        distances, motions, alignments = map(np.asarray, (distances, motions, alignments))
        expert_similarity = -distances
        distance_scale = float(np.median(distances))
        for temperature in (3.0, 5.0, 10.0):
            confidence = np.exp(-distances / (temperature * distance_scale))
            for alignment_floor in (0.0, 0.25, 0.50):
                alignment_gate = np.clip(
                    (alignments - alignment_floor) / (1.0 - alignment_floor), 0.0, 1.0
                )
                for motion_quantile in (0.50, 0.75):
                    motion_scale = float(np.quantile(lib_path[rows], motion_quantile))
                    motion_gate = np.clip(motions / max(motion_scale, 1e-6), 0.0, 1.0)
                    alpha = confidence * alignment_gate * motion_gate
                    values = {
                        "alpha_expert_similarity_correlation": v239.corr(alpha, expert_similarity),
                        "alpha_motion_correlation": v239.corr(alpha, motions),
                        "alpha_directional_alignment_correlation": v239.corr(alpha, alignments),
                    }
                    reports.append({
                        "library_path_quantile": quantile,
                        "library_rows": int(rows.size),
                        "temperature": temperature,
                        "alignment_floor": alignment_floor,
                        "motion_quantile": motion_quantile,
                        "motion_scale": motion_scale,
                        "alpha": v239.stats(alpha),
                        **values,
                        "selection_score": min(values.values()),
                    })
    valid = [item for item in reports if item["selection_score"] > 0]
    selected = max(valid, key=lambda item: item["selection_score"]) if valid else None
    report = {
        "format": "strict-track2-v240-alignment-gated-transport-analysis-v1",
        "source": "public v211 captures and declared public demonstration library",
        "post_grasp_queries": len(captures),
        "candidate_count": len(reports),
        "selection_rule": (
            "maximize the minimum alpha correlation with expert similarity, motion, and directional alignment; "
            "all three must be positive"
        ),
        "selected": selected,
        "top_candidates": sorted(valid, key=lambda item: item["selection_score"], reverse=True)[:10],
        "guards": {"policy_modified": False, "public_data_only": True,
                   "hidden_or_final_data": False, "real_submission": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
