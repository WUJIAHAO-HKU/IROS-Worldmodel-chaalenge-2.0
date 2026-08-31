#!/usr/bin/env python3
"""Quantify closed-loop visual degradation in tiled Track2 rollout videos.

This is a training-rollout diagnostic only.  It neither reads evaluation
outcomes nor authorizes a candidate.  Each 1792x1280 video is interpreted as
a 7x5 grid of 256px trajectories; empty padding cells are detected from the
first frame and excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


GRID_COLUMNS = 7
GRID_ROWS = 5
TILE_SIZE = 256
INNER = (slice(24, 248), slice(8, 248))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def split_tiles(frame: np.ndarray) -> list[np.ndarray]:
    expected = (GRID_ROWS * TILE_SIZE, GRID_COLUMNS * TILE_SIZE)
    if frame.shape[:2] != expected:
        raise RuntimeError(f"unexpected frame shape {frame.shape[:2]}, expected {expected}")
    return [
        frame[row * TILE_SIZE : (row + 1) * TILE_SIZE,
              column * TILE_SIZE : (column + 1) * TILE_SIZE]
        for row in range(GRID_ROWS)
        for column in range(GRID_COLUMNS)
    ]


def tile_metrics(tile: np.ndarray) -> dict[str, float]:
    image = tile[INNER]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    histogram = cv2.calcHist([gray], [0], None, [64], [0, 256]).ravel()
    probabilities = histogram / max(float(histogram.sum()), 1.0)
    probabilities = probabilities[probabilities > 0]
    return {
        "mean_luma": float(gray.mean()),
        "contrast": float(gray.std()),
        "laplacian_variance": float(laplacian.var()),
        "gradient_energy": float(np.mean(np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y))),
        "mean_saturation": float(hsv[..., 1].mean()),
        "entropy_bits": float(-(probabilities * np.log2(probabilities)).sum()),
        "near_black_fraction": float(np.mean(gray < 8)),
        "near_white_fraction": float(np.mean(gray > 247)),
    }


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(denominator, 1e-9))


def label_tile(tile: np.ndarray, label: str) -> np.ndarray:
    result = tile.copy()
    cv2.rectangle(result, (0, 0), (255, 25), (0, 0, 0), -1)
    cv2.putText(result, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (0, 255, 255), 1, cv2.LINE_AA)
    return result


def write_stage_sheet(path: Path, groups: list[tuple[str, list[np.ndarray]]]) -> None:
    rows = []
    for video_name, frames in groups:
        stage_indices = sorted({0, len(frames) // 4, len(frames) // 2,
                                (3 * len(frames)) // 4, len(frames) - 1})
        tiles_by_stage = [split_tiles(frames[index]) for index in stage_indices]
        for cell in range(GRID_ROWS * GRID_COLUMNS):
            first_inner = tiles_by_stage[0][cell][INNER]
            gray = cv2.cvtColor(first_inner, cv2.COLOR_BGR2GRAY)
            if float(gray.mean()) < 5.0 and float(gray.std()) < 5.0:
                continue
            row = np.concatenate([
                label_tile(stage[cell], f"{video_name} c{cell:02d} f{index:02d}")
                for stage, index in zip(tiles_by_stage, stage_indices)
            ], axis=1)
            rows.append(row)
    if not rows:
        raise RuntimeError("no active video cells")
    # Keep sheets readable and bounded: one image per rollout video is produced
    # by the caller, so at most 32 trajectory rows are stacked here.
    if not cv2.imwrite(str(path), np.concatenate(rows, axis=0)):
        raise RuntimeError(f"failed to write {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--visual-output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = sorted(args.videos.glob("*.mp4"))
    if len(paths) != 4:
        raise RuntimeError(f"expected exactly four rollout videos, found {len(paths)}")
    args.visual_output.mkdir(parents=True, exist_ok=True)

    trajectory_records = []
    video_records = []
    for path in paths:
        capture = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        metadata = {
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        capture.release()
        if len(frames) != 26:
            raise RuntimeError(f"{path}: expected 26 decoded frames, found {len(frames)}")
        write_stage_sheet(args.visual_output / f"{path.stem}_trajectories.jpg",
                          [(path.stem, frames)])

        tiled_frames = [split_tiles(frame) for frame in frames]
        active_cells = []
        for cell, tile in enumerate(tiled_frames[0]):
            gray = cv2.cvtColor(tile[INNER], cv2.COLOR_BGR2GRAY)
            if not (float(gray.mean()) < 5.0 and float(gray.std()) < 5.0):
                active_cells.append(cell)
        if len(active_cells) != 32:
            raise RuntimeError(f"{path}: expected 32 active cells, found {len(active_cells)}")

        for cell in active_cells:
            per_frame = [tile_metrics(tiles[cell]) for tiles in tiled_frames]
            first_image = tiled_frames[0][cell][INNER].astype(np.float32)
            initial_mae = [
                float(np.mean(np.abs(tiles[cell][INNER].astype(np.float32) - first_image)))
                for tiles in tiled_frames
            ]
            consecutive_mae = [0.0] + [
                float(np.mean(np.abs(tiled_frames[index][cell][INNER].astype(np.float32)
                                     - tiled_frames[index - 1][cell][INNER].astype(np.float32))))
                for index in range(1, len(tiled_frames))
            ]
            record = {
                "trajectory_index": len(trajectory_records),
                "video": path.name,
                "cell": cell,
                "per_frame": per_frame,
                "initial_frame_mae": initial_mae,
                "consecutive_frame_mae": consecutive_mae,
                "final_over_initial": {
                    key: ratio(per_frame[-1][key], per_frame[0][key])
                    for key in ("contrast", "laplacian_variance", "gradient_energy",
                                "mean_saturation", "entropy_bits")
                },
            }
            trajectory_records.append(record)
        video_records.append({
            "video": path.name,
            "sha256": sha256(path),
            "metadata": metadata,
            "active_cells": active_cells,
        })

    if len(trajectory_records) != 128:
        raise RuntimeError(f"expected 128 trajectories, found {len(trajectory_records)}")
    stages = [0, 6, 13, 19, 25]
    stage_summary = {}
    for frame_index in stages:
        stage_summary[str(frame_index)] = {
            key: summarize([record["per_frame"][frame_index][key]
                            for record in trajectory_records])
            for key in ("contrast", "laplacian_variance", "gradient_energy",
                        "mean_saturation", "entropy_bits", "near_black_fraction",
                        "near_white_fraction")
        }
        stage_summary[str(frame_index)]["initial_frame_mae"] = summarize([
            record["initial_frame_mae"][frame_index] for record in trajectory_records
        ])
        stage_summary[str(frame_index)]["consecutive_frame_mae"] = summarize([
            record["consecutive_frame_mae"][frame_index] for record in trajectory_records
        ])

    final_ratios = {
        key: summarize([record["final_over_initial"][key] for record in trajectory_records])
        for key in ("contrast", "laplacian_variance", "gradient_energy",
                    "mean_saturation", "entropy_bits")
    }
    report = {
        "format": "strict-track2-v327-training-rollout-recursive-quality-diagnostic-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "post-hoc exploratory diagnostic of training-rollout video only",
        "authorizing": False,
        "videos": video_records,
        "trajectory_count": len(trajectory_records),
        "sampled_frame_indices": stages,
        "stage_summary": stage_summary,
        "final_over_initial_ratio_summary": final_ratios,
        "trajectory_records": trajectory_records,
        "interpretation_contract": {
            "lower_laplacian_or_gradient_ratio_indicates_loss_of_high_frequency_detail": True,
            "higher_initial_mae_indicates_visual_drift_not_necessarily_task_progress": True,
            "no_metric_is_a_candidate_selection_gate": True,
        },
        "guards": {
            "training_rollout_only": True,
            "public_evaluation_outcomes_read": False,
            "hidden_or_final_data_read": False,
            "real_submission": False,
        },
    }
    # Ensure JSON contains no NaN/Inf values.
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded)
    concise = {
        "trajectory_count": report["trajectory_count"],
        "stage_summary": report["stage_summary"],
        "final_over_initial_ratio_summary": report["final_over_initial_ratio_summary"],
        "guards": report["guards"],
    }
    print(json.dumps(concise, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
