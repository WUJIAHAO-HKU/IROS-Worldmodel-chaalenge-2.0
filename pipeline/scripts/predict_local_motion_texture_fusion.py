#!/usr/bin/env python3
"""Apply a trained local motion/texture fusion model to aligned parent caches."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from wam_pipeline.local_fusion import LocalMotionTextureFusion


def load_cache(path: Path) -> tuple[np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as cache:
        return cache["prediction"], [str(name) for name in cache["windows"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--autoregressive-cache", required=True)
    parser.add_argument("--direct-flow-cache", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    first, names = load_cache(Path(args.autoregressive_cache))
    second, second_names = load_cache(Path(args.direct_flow_cache))
    if names != second_names or first.shape != second.shape:
        raise ValueError("parent caches are not aligned")
    checkpoint = Path(args.checkpoint_dir)
    config = np.load(checkpoint / "local_fusion_config.npz", allow_pickle=False)
    model = LocalMotionTextureFusion(int(config["base_channels"]), float(config["residual_scale"]))
    state = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True)
    if state.get("format") != "track2-local-motion-texture-fusion-v1":
        raise ValueError("unsupported local fusion checkpoint")
    model.load_state_dict(state["state_dict"], strict=True)
    device = torch.device(args.device)
    model = model.to(device).eval()
    normalization = np.load(checkpoint / "action_normalization.npz", allow_pickle=False)
    mean = torch.from_numpy(np.asarray(normalization["mean"], dtype=np.float32)).to(device)
    std = torch.from_numpy(np.asarray(normalization["std"], dtype=np.float32)).to(device)
    predictions = np.empty_like(first)
    alpha_sum = residual_sum = alpha_count = 0.0
    for start in range(0, len(names), args.batch_size):
        stop = min(start + args.batch_size, len(names))
        contexts, actions = [], []
        for name in names[start:stop]:
            with np.load(Path(args.windows) / name, allow_pickle=False) as window:
                contexts.append(window["context_frames"][-1:])
                actions.append(window["future_actions"])
        context = torch.from_numpy(np.stack(contexts)).permute(0, 1, 4, 2, 3).to(device).float().div(255.0)
        first_batch = torch.from_numpy(first[start:stop].copy()).permute(0, 1, 4, 2, 3).to(device).float().div(255.0)
        second_batch = torch.from_numpy(second[start:stop].copy()).permute(0, 1, 4, 2, 3).to(device).float().div(255.0)
        action = (torch.from_numpy(np.stack(actions)).to(device).float() - mean) / std
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction, alpha, residual = model(context, first_batch, second_batch, action)
        predictions[start:stop] = prediction.clamp(0, 1).mul(255).round().to(torch.uint8).permute(0, 1, 3, 4, 2).cpu().numpy()
        alpha_sum += float(alpha.float().sum())
        residual_sum += float(residual.float().abs().sum())
        alpha_count += alpha.numel()
        if stop % 100 == 0 or stop == len(names):
            print(json.dumps({"event": "fusion_prediction_progress", "completed": stop, "total": len(names)}), flush=True)
    output_cache = Path(args.output_cache)
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    temporary_cache = output_cache.with_suffix(output_cache.suffix + f".tmp.{os.getpid()}")
    with temporary_cache.open("wb") as handle:
        np.savez_compressed(handle, prediction=predictions, windows=np.asarray(names))
    os.replace(temporary_cache, output_cache)
    result = {"format": "track2-local-motion-texture-fusion-prediction-v1", "checkpoint_dir": str(checkpoint.resolve()), "sample_count": len(names), "alpha_mean": alpha_sum / alpha_count, "residual_abs_mean_normalized": residual_sum / (alpha_count * 3.0), "output_cache": str(output_cache.resolve())}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
