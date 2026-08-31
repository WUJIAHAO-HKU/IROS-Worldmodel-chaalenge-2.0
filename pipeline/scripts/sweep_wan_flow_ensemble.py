#!/usr/bin/env python3
"""Select an RGB ensemble of two audited Track 2 action-conditioned models.

Both members receive exactly the official 5 context frames, four history
actions, and eight future actions.  Predictions are generated once per model
on the same windows, then only the final RGB blend weight is swept on CPU.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wam_pipeline.data import load_window_npz
from wam_pipeline.direct_flow_unet_runtime import Track2DirectFlowUNet
from wam_pipeline.track2_wan_runtime import Track2WanRuntime


def evenly_spaced(paths: list[Path], count: int) -> list[Path]:
    if count < 1:
        raise ValueError("samples must be positive")
    if count >= len(paths):
        return paths
    return [paths[index] for index in np.linspace(0, len(paths) - 1, count, dtype=np.int64)]


def motion_score(context: np.ndarray, target: np.ndarray) -> float:
    video = np.concatenate((context[-1:], target), axis=0).astype(np.float32)
    return float(np.abs(video[1:] - video[:-1]).mean() / 255.0)


def evaluate(prediction: np.ndarray, target: np.ndarray, high_motion: np.ndarray) -> dict[str, object]:
    per_frame = np.abs(prediction.astype(np.float32) - target.astype(np.float32)).mean(axis=(0, 2, 3, 4))
    per_window_frame = np.abs(prediction.astype(np.float32) - target.astype(np.float32)).mean(axis=(2, 3, 4))
    result: dict[str, object] = {
        "mae_by_prediction_frame": [float(value) for value in per_frame],
        "mae_mean": float(per_frame.mean()),
        "max_window_prediction_frame_mae": float(per_window_frame.max()),
    }
    if high_motion.any():
        result["high_motion_mae_mean"] = float(per_window_frame[high_motion].mean())
    else:
        result["high_motion_mae_mean"] = None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep deterministic Wan/direct-flow RGB blend weights.")
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--wan-checkpoint", required=True)
    parser.add_argument("--direct-flow-checkpoint", required=True)
    parser.add_argument("--wan-base-model", required=True)
    parser.add_argument("--split", choices=("validation", "local-test"), default="validation")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wan-inference-steps", type=int, default=45)
    parser.add_argument("--wan-inference-solver", choices=("euler", "heun"), default="euler")
    parser.add_argument(
        "--wan-predictions-cache",
        help="Optional .npz cache from evaluate_ivideogpt64.py; ordered windows are verified before use.",
    )
    parser.add_argument("--weights", type=float, nargs="+", default=(0.0, 0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--high-motion-threshold", type=float, default=0.04)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.samples < 1 or args.wan_inference_steps < 1 or not args.weights:
        raise SystemExit("samples, Wan inference steps, and at least one weight must be positive")
    if any(not 0 <= weight <= 1 for weight in args.weights):
        raise SystemExit("every --weights value must be in [0, 1]")

    split = json.loads(Path(args.split_manifest).read_text())
    key = "validation_episodes" if args.split == "validation" else "local_test_episodes"
    paths = [
        path
        for episode in split[key]
        for path in sorted(Path(args.windows).glob(f"episode{episode}_*.npz"))
    ]
    selected = evenly_spaced(paths, args.samples)
    windows = [load_window_npz(path) for path in selected]
    target = np.stack([window.target_frames for window in windows])
    high_motion = np.asarray(
        [motion_score(window.context_frames, window.target_frames) >= args.high_motion_threshold for window in windows]
    )

    cache = Path(args.wan_predictions_cache) if args.wan_predictions_cache else None
    if cache and cache.is_file():
        loaded = np.load(cache, allow_pickle=False)
        if set(loaded.files) != {"prediction", "windows"}:
            raise SystemExit("Wan prediction cache must contain prediction and windows arrays")
        wan_prediction = loaded["prediction"]
        expected_shape = target.shape
        if wan_prediction.shape != expected_shape or wan_prediction.dtype != np.uint8:
            raise SystemExit(f"Wan prediction cache must be uint8 {expected_shape}, got {wan_prediction.shape} {wan_prediction.dtype}")
        if loaded["windows"].tolist() != [path.name for path in selected]:
            raise SystemExit("Wan prediction cache windows do not match this sweep")
    else:
        wan = Track2WanRuntime(
            args.wan_checkpoint,
            base_model=args.wan_base_model,
            device=args.device,
            inference_steps=args.wan_inference_steps,
            inference_solver=args.wan_inference_solver,
        )
        wan_prediction = np.stack(
            [
                wan.predict(window.context_frames, window.history_actions, window.future_actions, args.seed, None)
                for window in windows
            ]
        )
        del wan
        gc.collect()
        if args.device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            with cache.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    prediction=wan_prediction,
                    windows=np.asarray([path.name for path in selected]),
                )

    direct_flow = Track2DirectFlowUNet(args.direct_flow_checkpoint, args.device)
    flow_prediction = np.stack(
        [
            direct_flow.predict(window.context_frames, window.history_actions, window.future_actions, args.seed, None)
            for window in windows
        ]
    )
    del direct_flow
    gc.collect()
    if args.device.startswith("cuda"):
        import torch

        torch.cuda.empty_cache()

    candidates = []
    for wan_weight in args.weights:
        blend = np.rint(wan_prediction.astype(np.float32) * wan_weight + flow_prediction.astype(np.float32) * (1.0 - wan_weight))
        metrics = evaluate(blend.clip(0, 255).astype(np.uint8), target, high_motion)
        candidates.append({"wan_weight": float(wan_weight), "direct_flow_weight": float(1.0 - wan_weight), **metrics})
    best = min(candidates, key=lambda item: float(item["mae_mean"]))
    result = {
        "format": "track2-wan-flow-ensemble-sweep-v1",
        "split": args.split,
        "sample_count": len(selected),
        "windows": [path.name for path in selected],
        "wan_checkpoint": str(Path(args.wan_checkpoint).resolve()),
        "direct_flow_checkpoint": str(Path(args.direct_flow_checkpoint).resolve()),
        "wan_inference": {"steps": args.wan_inference_steps, "solver": args.wan_inference_solver},
        "high_motion_sample_count": int(high_motion.sum()),
        "candidates": candidates,
        "selected": best,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
