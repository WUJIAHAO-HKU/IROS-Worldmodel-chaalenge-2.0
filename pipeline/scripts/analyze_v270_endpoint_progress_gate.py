#!/usr/bin/env python3
"""Measure absolute expert-endpoint progress for the public v254 audit queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def describe(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        key: float(value)
        for key, value in zip(
            ("min", "q10", "q25", "median", "q75", "q90", "max"),
            np.quantile(values, (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)),
        )
    }


def correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    if len(left) < 2 or left.std() == 0 or right.std() == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def endpoint_metrics(history, future, reference, scale) -> dict:
    query_start = np.asarray(history[-1, 7:13], dtype=np.float64)
    query_end = np.asarray(future[-1, 7:13], dtype=np.float64)
    reference_start = np.asarray(reference[-9, 7:13], dtype=np.float64)
    reference_end = np.asarray(reference[-1, 7:13], dtype=np.float64)
    query_delta = query_end - query_start
    reference_delta = reference_end - reference_start
    start_distance = float(np.mean(((query_start - reference_start) / scale) ** 2))
    endpoint_distance = float(np.mean(((query_end - reference_end) / scale) ** 2))
    reference_norm = float(np.linalg.norm(reference_delta))
    query_norm = float(np.linalg.norm(query_delta))
    denominator = max(reference_norm**2, 1e-12)
    projection = float(np.dot(query_delta, reference_delta) / denominator)
    orthogonal = query_delta - projection * reference_delta
    return {
        "start_distance": start_distance,
        "endpoint_distance": endpoint_distance,
        "absolute_progress": start_distance - endpoint_distance,
        "relative_progress": (start_distance - endpoint_distance) / max(start_distance, 1e-12),
        "projection_ratio": projection,
        "motion_scale_ratio": query_norm / max(reference_norm, 1e-12),
        "orthogonal_ratio": float(np.linalg.norm(orthogonal) / max(reference_norm, 1e-12)),
        "endpoint_l2": float(np.linalg.norm(query_end - reference_end)),
    }


def summarize(rows: list[dict]) -> dict:
    keys = (
        "start_distance",
        "endpoint_distance",
        "absolute_progress",
        "relative_progress",
        "projection_ratio",
        "motion_scale_ratio",
        "orthogonal_ratio",
        "endpoint_l2",
    )
    if not rows:
        return {"count": 0, **{key: None for key in keys}}
    return {"count": len(rows), **{key: describe([row[key] for row in rows]) for key in keys}}


def summarize_terminal(rows: list[dict]) -> dict:
    keys = (
        "terminal_start_distance",
        "terminal_endpoint_distance",
        "terminal_absolute_progress",
        "terminal_relative_progress",
        "terminal_endpoint_l2",
    )
    if not rows:
        return {"count": 0, **{key: None for key in keys}}
    return {"count": len(rows), **{key: describe([row[key] for row in rows]) for key in keys}}


def visual_descriptor(frame: np.ndarray) -> np.ndarray:
    if frame.shape != (256, 256, 3):
        raise ValueError(f"expected 256x256 RGB context, got {frame.shape}")
    return frame.astype(np.float32).reshape(16, 16, 16, 16, 3).mean((1, 3)).reshape(-1) / 255.0


def parse_name(name: str) -> tuple[int, int]:
    episode, start = Path(name).stem.split("_")
    return int(episode[7:]), int(start)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--capture-details", type=Path)
    parser.add_argument("--success-windows", required=True, type=Path)
    parser.add_argument("--failure-windows", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    analysis = json.loads(args.analysis.read_text())
    if not analysis["guards"]["public_data_only"] or analysis["guards"]["hidden_or_final_data"]:
        raise RuntimeError("input analysis violates public-only guard")
    with np.load(args.library, allow_pickle=False) as values:
        normalized = values["action"].astype(np.float32)
        mean = values["normalization_mean"].astype(np.float32)
        std = values["normalization_std"].astype(np.float32)
        episode_ids = values["episode_id"].astype(np.int64)
        paths = values["path"].astype(str)
        visual = values["visual"].astype(np.float32)
    raw = normalized.reshape(-1, 12, 14) * std + mean
    scale = np.maximum(std[7:13].astype(np.float64), 1e-6)
    clean = np.flatnonzero(episode_ids >= 20000)
    episode_rows = {}
    for row in clean:
        start = int(Path(paths[row]).stem.split("_")[1])
        episode_rows.setdefault(int(episode_ids[row]), {})[start] = int(row)
    terminal_rows = {}
    for episode, lookup in episode_rows.items():
        starts = np.asarray(sorted(lookup), dtype=np.int64)
        late = starts[starts >= 112]
        chosen = int(late[-1] if len(late) else starts[-1])
        terminal_rows[episode] = lookup[chosen]

    def terminal_for_context(context) -> int:
        distance = ((visual[clean] - visual_descriptor(context[-1])) ** 2).mean(1)
        visual_row = int(clean[int(np.argmin(distance))])
        return int(terminal_rows[int(episode_ids[visual_row])])

    def with_terminal(history, future, context, base) -> dict:
        result = endpoint_metrics(history, future, raw[base], scale)
        target = terminal_for_context(context)
        terminal = endpoint_metrics(history, future, raw[target], scale)
        result["terminal_row"] = target
        result.update({f"terminal_{key}": value for key, value in terminal.items()})
        return result

    capture_files = sorted(args.capture.glob("rollout_*.npz"))
    capture_queries = {}
    for file_index, path in enumerate(capture_files):
        with np.load(path, allow_pickle=False) as values:
            contexts = values["context_frames"].copy()
            histories = values["history_actions"].astype(np.float32)
            futures = values["future_actions"].astype(np.float32)
        for action_index, (context, history, future) in enumerate(zip(contexts, histories, futures)):
            capture_queries[(file_index, action_index)] = (context, history, future)
    capture_rows = []
    runtime_bases = None
    if args.capture_details is not None:
        with np.load(args.capture_details, allow_pickle=False) as values:
            runtime_bases = values["base"].astype(np.int64)
            runtime_delta_ratio = values["delta_ratio"].astype(np.float64)
        if len(runtime_bases) != len(analysis["capture_rows"]):
            raise RuntimeError("capture details count mismatch")
    for row_index, original in enumerate(analysis["capture_rows"]):
        context, history, future = capture_queries[(original["file"], original["action"])]
        runtime_base = int(runtime_bases[row_index]) if runtime_bases is not None else int(original["base"])
        if runtime_bases is not None and not np.isclose(runtime_delta_ratio[row_index], original["delta_ratio"]):
            raise RuntimeError("capture details order mismatch")
        capture_rows.append(
            {
                **original,
                "delta_retrieval_base": int(original["base"]),
                "base": runtime_base,
                **with_terminal(history, future, context, runtime_base),
            }
        )

    indexes = {}
    for split, root in (("success", args.success_windows), ("failure", args.failure_windows)):
        indexes[split] = {parse_name(path.name): path for path in root.glob("episode*_*.npz")}
    long_rows = []
    for original in analysis["long_rows"]:
        episode, start = parse_name(original["path"])
        initial = indexes[original["split"]][(episode, start)]
        future_path = indexes[original["split"]][(episode, start + 8 * original["chunk"])]
        with np.load(initial, allow_pickle=False) as values:
            context = values["context_frames"].copy()
            history = values["history_actions"].astype(np.float32)
        with np.load(future_path, allow_pickle=False) as values:
            future = values["future_actions"].astype(np.float32)
        long_rows.append({**original, **with_terminal(history, future, context, original["base"])})

    clean_capture = [row for row in capture_rows if row["delta_ratio"] <= 0.01]
    ood_capture = [row for row in capture_rows if row["delta_ratio"] >= 0.8]
    mid_capture = [row for row in capture_rows if 0.01 < row["delta_ratio"] < 0.8]
    report = {
        "format": "strict-track2-v270-endpoint-progress-gate-analysis-v1",
        "source_analysis": str(args.analysis),
        "right_joint_scale": scale.tolist(),
        "capture": {
            "all": summarize(capture_rows),
            "all_visual_terminal": summarize_terminal(capture_rows),
            "clean_regime": summarize(clean_capture),
            "clean_regime_visual_terminal": summarize_terminal(clean_capture),
            "mid_regime": summarize(mid_capture),
            "mid_regime_visual_terminal": summarize_terminal(mid_capture),
            "ood_regime": summarize(ood_capture),
            "ood_regime_visual_terminal": summarize_terminal(ood_capture),
            "ood_reward_correlations": {
                "negative_endpoint_distance": correlation(
                    [row["terminal_reward"] for row in ood_capture],
                    [-row["endpoint_distance"] for row in ood_capture],
                ),
                "relative_progress": correlation(
                    [row["terminal_reward"] for row in ood_capture],
                    [row["relative_progress"] for row in ood_capture],
                ),
                "negative_orthogonal_ratio": correlation(
                    [row["terminal_reward"] for row in ood_capture],
                    [-row["orthogonal_ratio"] for row in ood_capture],
                ),
                "negative_visual_terminal_endpoint_distance": correlation(
                    [row["terminal_reward"] for row in ood_capture],
                    [-row["terminal_endpoint_distance"] for row in ood_capture],
                ),
                "visual_terminal_relative_progress": correlation(
                    [row["terminal_reward"] for row in ood_capture],
                    [row["terminal_relative_progress"] for row in ood_capture],
                ),
            },
        },
        "long_first_trigger": {
            split: {
                "retrieved_base": summarize([row for row in long_rows if row["split"] == split]),
                "visual_terminal": summarize_terminal([row for row in long_rows if row["split"] == split]),
            }
            for split in ("success", "failure")
        },
        "capture_rows": capture_rows,
        "long_rows": long_rows,
        "guards": {
            "public_data_only": True,
            "policy_modified": False,
            "runtime_uses_reward": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"capture": report["capture"], "long_first_trigger": report["long_first_trigger"]}, indent=2))


if __name__ == "__main__":
    main()
