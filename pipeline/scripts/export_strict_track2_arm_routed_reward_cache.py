#!/usr/bin/env python3
"""Export direct arm-routed AR rollouts for frozen Track 2 reward auditing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_strict_track2_autoregressive_candidate import Windows, load_model, rollout


def as_frames(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(value).permute(0, 3, 1, 2).to(device).float().div(255)


def predict(model, mean, std, context, history, future) -> np.ndarray:
    context_tensor = as_frames(context, mean.device)[None]
    history_tensor = (torch.from_numpy(history).to(mean.device)[None] - mean) / std
    future_tensor = (torch.from_numpy(future).to(mean.device)[None] - mean) / std
    prediction = rollout(model, context_tensor, history_tensor, future_tensor)[0]
    return (
        prediction.mul(255).round().clamp(0, 255)
        .byte().permute(0, 2, 3, 1).cpu().numpy()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--episodes-key", default="validation_episodes")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--left-candidate", required=True)
    parser.add_argument("--right-candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--min-start", type=int)
    parser.add_argument("--max-start", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    split = json.loads(Path(args.split_manifest).read_text())
    dataset = Windows(
        Path(args.windows), split[args.episodes_key],
        min_start=args.min_start, max_start=args.max_start,
    )
    indices = list(range(len(dataset)))
    if args.max_windows and args.max_windows < len(indices):
        indices = np.linspace(0, len(indices) - 1, args.max_windows, dtype=int).tolist()

    device = torch.device(args.device)
    baseline = load_model(Path(args.baseline), device)
    left = load_model(Path(args.left_candidate), device)
    right = load_model(Path(args.right_candidate), device)
    records = {
        key: [] for key in (
            "context_last", "target", "baseline", "candidate", "arm_right",
            "capture_success", "path", "synthetic_seed", "start",
        )
    }
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        for position, index in enumerate(indices, 1):
            context, history, future, target, arm_right, capture_success = dataset[index]
            candidate = right if arm_right else left
            baseline_prediction = predict(*baseline, context, history, future)
            candidate_prediction = predict(*candidate, context, history, future)
            path = dataset.paths[index]
            with np.load(path, allow_pickle=False) as values:
                synthetic_seed = int(values["synthetic_seed"]) if "synthetic_seed" in values.files else -1
                start = int(values["start"]) if "start" in values.files else dataset.starts[index]
            values = {
                "context_last": context[-1],
                "target": target,
                "baseline": baseline_prediction,
                "candidate": candidate_prediction,
                "arm_right": np.asarray(arm_right, dtype=np.bool_),
                "capture_success": np.asarray(capture_success, dtype=np.bool_),
                "path": path.name,
                "synthetic_seed": np.asarray(synthetic_seed, dtype=np.int64),
                "start": np.asarray(start, dtype=np.int64),
            }
            for key, value in values.items():
                records[key].append(value)
            if position == 1 or position % 8 == 0 or position == len(indices):
                print(json.dumps({"exported": position, "total": len(indices)}), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **{
        key: np.asarray(value) if key == "path" else np.stack(value)
        for key, value in records.items()
    })
    manifest = {
        "format": "strict-track2-direct-arm-routed-reward-cache-v1",
        "output": str(output.resolve()),
        "window_count": len(indices),
        "frame_count_per_window": 8,
        "split_manifest": str(Path(args.split_manifest).resolve()),
        "episodes_key": args.episodes_key,
        "min_start": args.min_start,
        "max_start": args.max_start,
        "baseline": str(Path(args.baseline).resolve()),
        "left_candidate": str(Path(args.left_candidate).resolve()),
        "right_candidate": str(Path(args.right_candidate).resolve()),
        "routing": "raw_action_delta_argmax",
        "paths": [dataset.paths[index].name for index in indices],
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
