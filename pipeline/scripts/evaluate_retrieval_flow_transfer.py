#!/usr/bin/env python3
"""Prototype nearest-trajectory motion transfer without learned RGB synthesis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def image_feature(context: np.ndarray) -> np.ndarray:
    last = cv2.resize(context[-1], (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    first = cv2.resize(context[0], (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    gray = cv2.cvtColor(last, cv2.COLOR_RGB2GRAY)
    edge = cv2.Laplacian(gray, cv2.CV_32F)
    return np.concatenate((last.ravel(), (last - first).ravel(), edge.ravel()))


def action_feature(history: np.ndarray, future: np.ndarray) -> np.ndarray:
    return np.concatenate((history, future), axis=0).astype(np.float32).ravel()


def transfer(source_last: np.ndarray, source_future: np.ndarray, query_last: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = query_last.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    source_gray = cv2.cvtColor(source_last, cv2.COLOR_RGB2GRAY)
    warped_values, repaired_values = [], []
    for target in source_future:
        target_gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
        backward = cv2.calcOpticalFlowFarneback(target_gray, source_gray, None, 0.5, 5, 21, 4, 7, 1.5, 0)
        map_x, map_y = grid_x + backward[..., 0], grid_y + backward[..., 1]
        query_warp = cv2.remap(query_last, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        source_warp = cv2.remap(source_last, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        residual = target.astype(np.float32) - source_warp.astype(np.float32)
        repaired = np.clip(query_warp.astype(np.float32) + residual, 0, 255).round().astype(np.uint8)
        warped_values.append(query_warp)
        repaired_values.append(repaired)
    return np.stack(warped_values), np.stack(repaired_values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--train-cache", required=True, help="Only its ordered window names are used.")
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--action-weights", default="0,0.01,0.03,0.1,0.3,1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    windows = Path(args.windows)
    with np.load(args.train_cache, allow_pickle=False) as cache:
        train_names = cache["windows"].astype(str)
    baseline = np.load(Path(args.baseline_dir) / "prediction.npy", mmap_mode="r")
    validation_names = np.load(Path(args.baseline_dir) / "windows.npy", allow_pickle=False).astype(str)
    selected_indices = np.linspace(0, len(validation_names) - 1, min(args.samples, len(validation_names)), dtype=np.int64)
    train_image, train_action = [], []
    for index, name in enumerate(train_names):
        with np.load(windows / name, allow_pickle=False) as value:
            train_image.append(image_feature(value["context_frames"]))
            train_action.append(action_feature(value["history_actions"], value["future_actions"]))
        if (index + 1) % 300 == 0:
            print(json.dumps({"event": "index", "completed": index + 1, "total": len(train_names)}), flush=True)
    train_image = np.stack(train_image)
    train_action = np.stack(train_action)
    action_mean, action_std = train_action.mean(0), train_action.std(0).clip(1e-4)
    train_action = (train_action - action_mean) / action_std
    weights = [float(value) for value in args.action_weights.split(",")]
    methods = ["baseline"] + [f"replay_w{w:g}" for w in weights] + [f"delta_w{w:g}" for w in weights]
    sums = {name: 0.0 for name in methods}
    count = 0
    chosen: dict[float, list[int]] = {weight: [] for weight in weights}
    queries = []
    for completed, query_index in enumerate(selected_indices, start=1):
        name = validation_names[query_index]
        with np.load(windows / name, allow_pickle=False) as query:
            context = query["context_frames"]
            target = query["target_frames"].astype(np.float32)
            q_image = image_feature(context)
            q_action = (action_feature(query["history_actions"], query["future_actions"]) - action_mean) / action_std
        image_distance = np.mean((train_image - q_image) ** 2, axis=1)
        action_distance = np.mean((train_action - q_action) ** 2, axis=1)
        image_scale = np.median(image_distance).clip(1e-8)
        action_scale = np.median(action_distance).clip(1e-8)
        sums["baseline"] += float(np.abs(baseline[query_index].astype(np.float32) - target).sum(dtype=np.float64))
        for weight in weights:
            source_index = int(np.argmin(image_distance / image_scale + weight * action_distance / action_scale))
            chosen[weight].append(source_index)
            with np.load(windows / train_names[source_index], allow_pickle=False) as source:
                replay = source["target_frames"].astype(np.float32)
                delta = np.clip(context[-1:].astype(np.float32) + replay - source["context_frames"][-1:].astype(np.float32), 0, 255)
            sums[f"replay_w{weight:g}"] += float(np.abs(replay - target).sum(dtype=np.float64))
            sums[f"delta_w{weight:g}"] += float(np.abs(delta - target).sum(dtype=np.float64))
        count += target.size
        queries.append((int(query_index), name))
        if completed % 16 == 0:
            print(json.dumps({"event": "retrieval", "completed": completed, "total": len(selected_indices)}), flush=True)
    mae = {name: value / count for name, value in sums.items()}
    best_key = min((name for name in mae if name != "baseline"), key=mae.get)
    best_weight = float(best_key.rsplit("w", 1)[1])
    flow_sums = {"flow_warp": 0.0, "flow_repaired": 0.0}
    for completed, ((query_index, name), source_index) in enumerate(zip(queries, chosen[best_weight]), start=1):
        with np.load(windows / name, allow_pickle=False) as query, np.load(windows / train_names[source_index], allow_pickle=False) as source:
            target = query["target_frames"].astype(np.float32)
            warp, repaired = transfer(source["context_frames"][-1], source["target_frames"], query["context_frames"][-1])
        flow_sums["flow_warp"] += float(np.abs(warp.astype(np.float32) - target).sum(dtype=np.float64))
        flow_sums["flow_repaired"] += float(np.abs(repaired.astype(np.float32) - target).sum(dtype=np.float64))
        if completed % 16 == 0:
            print(json.dumps({"event": "flow", "completed": completed, "total": len(queries)}), flush=True)
    mae.update({name: value / count for name, value in flow_sums.items()})
    base = mae["baseline"]
    result = {
        "format": "track2-retrieval-flow-transfer-prototype-v1",
        "sample_count": len(selected_indices),
        "train_candidate_count": len(train_names),
        "best_retrieval_method": best_key,
        "metrics": {name: {"rgb_mae": value, "relative_improvement_over_baseline_percent": 100.0 * (base - value) / base} for name, value in mae.items()},
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
