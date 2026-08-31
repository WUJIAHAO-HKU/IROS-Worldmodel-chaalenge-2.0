#!/usr/bin/env python3
"""Produce the standard open-loop report directly from an audited prediction cache."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np


def checkpoint_artifact_sha256(checkpoint_dir: Path, backend: str) -> str | None:
    """Identify a packaged cache source with the same hash used by the strict gate."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    if backend == "local-motion-texture-fusion":
        from wam_pipeline.local_fusion_runtime import ensemble_artifact_sha256
    elif backend == "structure-gated-local-fusion":
        from wam_pipeline.structure_refiner_runtime import ensemble_artifact_sha256
    else:
        return None

    return ensemble_artifact_sha256(checkpoint_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--split", choices=("train", "validation", "local-test"), required=True)
    parser.add_argument("--prediction-cache", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--backend", default="local-motion-texture-fusion")
    parser.add_argument("--high-motion-threshold", type=float, default=0.04)
    parser.add_argument("--accept-mae", type=float, default=2.5)
    parser.add_argument("--worst-windows", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with np.load(args.prediction_cache, allow_pickle=False) as cache:
        prediction = cache["prediction"]
        names = [str(name) for name in cache["windows"]]
    manifest = json.loads(Path(args.split_manifest).read_text())
    allowed = {int(value) for value in manifest[args.split.replace("-", "_") + "_episodes"]}
    model_maes, copy_maes, motion_scores = [], [], []
    for index, name in enumerate(names):
        match = re.fullmatch(r"episode(\d+)_\d+\.npz", name)
        if match is None or int(match.group(1)) not in allowed:
            raise ValueError(f"cache window {name!r} is outside split {args.split!r}")
        with np.load(Path(args.windows) / name, allow_pickle=False) as window:
            context = window["context_frames"]
            target = window["target_frames"].astype(np.float32)
        model_maes.append(np.abs(prediction[index].astype(np.float32) - target).mean(axis=(1, 2, 3)))
        copy_maes.append(np.abs(context[-1:].astype(np.float32) - target).mean(axis=(1, 2, 3)))
        previous = np.concatenate([context[-1:], target[:-1].astype(np.uint8)], axis=0).astype(np.float32)
        motion_scores.append(float(np.abs(target - previous).mean() / 255.0))
    per_window = np.asarray(model_maes, dtype=np.float32)
    copy = np.asarray(copy_maes, dtype=np.float32)
    motion = np.asarray(motion_scores, dtype=np.float64)
    high = motion >= args.high_motion_threshold
    means, peaks = per_window.mean(axis=1), per_window.max(axis=1)
    acceptance = {
        "threshold_mae_0_255": args.accept_mae,
        "window_count": len(names),
        "windows_passing_mean_over_8_frames": int((means < args.accept_mae).sum()),
        "windows_passing_all_8_prediction_frames": int((peaks < args.accept_mae).sum()),
        "max_window_mean_over_8_frames": float(means.max()),
        "max_window_prediction_frame_mae": float(peaks.max()),
        "p50_window_mean_over_8_frames": float(np.quantile(means, 0.50)),
        "p90_window_mean_over_8_frames": float(np.quantile(means, 0.90)),
        "p99_window_mean_over_8_frames": float(np.quantile(means, 0.99)),
        "all_windows_and_frames_pass": bool((peaks < args.accept_mae).all()),
    }
    order = np.argsort(-peaks)[: args.worst_windows]
    window_acceptance = [{"window": name, "mean_mae_over_8_frames": float(means[index]), "max_prediction_frame_mae": float(peaks[index]), "mae_by_prediction_frame": per_window[index].tolist(), "passes_all_prediction_frames": bool(peaks[index] < args.accept_mae)} for index, name in enumerate(names)]
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    result = {
        "format": "track2-open-loop-eval-v1",
        "split": args.split,
        "episodes": sorted(allowed),
        "sample_count": len(names),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_model_sha256": checkpoint_artifact_sha256(checkpoint_dir, args.backend),
        "backend": args.backend,
        "wan_inference": None,
        "ensemble": None,
        "model_mae_by_prediction_frame": per_window.mean(axis=0).tolist(),
        "model_mae_mean": float(per_window.mean()),
        "copy_last_mae_by_prediction_frame": copy.mean(axis=0).tolist(),
        "copy_last_mae_mean": float(copy.mean()),
        "high_motion_threshold": args.high_motion_threshold,
        "high_motion_sample_count": int(high.sum()),
        "high_motion_model_mae_by_prediction_frame": per_window[high].mean(axis=0).tolist() if high.any() else None,
        "high_motion_model_mae_mean": float(per_window[high].mean()) if high.any() else None,
        "high_motion_copy_last_mae_by_prediction_frame": copy[high].mean(axis=0).tolist() if high.any() else None,
        "high_motion_copy_last_mae_mean": float(copy[high].mean()) if high.any() else None,
        "action_perturbation_mean_absolute_pixel_delta": None,
        "hard_acceptance": acceptance,
        "worst_validation_windows": [{"window": names[index], "motion_score": float(motion[index]), "mean_mae_over_8_frames": float(means[index]), "max_prediction_frame_mae": float(peaks[index]), "mae_by_prediction_frame": per_window[index].tolist()} for index in order],
        "worst_window_by_prediction_frame": [{"prediction_frame": frame, "window": names[int(np.argmax(per_window[:, frame]))], "mae": float(per_window[:, frame].max()), "motion_score": float(motion[int(np.argmax(per_window[:, frame]))])} for frame in range(8)],
        "failure_gifs": [],
        "window_acceptance": window_acceptance,
        "prediction_cache": str(Path(args.prediction_cache).resolve()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), "model_mae_mean": result["model_mae_mean"], "high_motion_model_mae_mean": result["high_motion_model_mae_mean"], "hard_acceptance": acceptance}, indent=2))


if __name__ == "__main__":
    main()
