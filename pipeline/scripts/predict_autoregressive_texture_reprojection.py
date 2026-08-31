#!/usr/bin/env python3
"""Apply a trained explicit texture-reprojection parent to an AR cache."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from wam_pipeline.autoregressive_texture_reprojection import AutoregressiveTextureReprojection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--autoregressive-cache", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with np.load(args.autoregressive_cache, allow_pickle=False) as cache:
        ar = cache["prediction"]
        names = [str(value) for value in cache["windows"]]
    checkpoint = Path(args.checkpoint_dir)
    config = np.load(checkpoint / "autoregressive_texture_reprojection_config.npz", allow_pickle=False)
    model = AutoregressiveTextureReprojection(int(config["base_channels"]), float(config["residual_scale"]))
    state = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True)
    if state.get("format") != "track2-autoregressive-texture-reprojection-v1":
        raise ValueError("unsupported texture-reprojection checkpoint")
    model.load_state_dict(state["state_dict"], strict=True)
    device = torch.device(args.device)
    model = model.to(device).eval()
    normalization = np.load(checkpoint / "action_normalization.npz", allow_pickle=False)
    mean = torch.from_numpy(np.asarray(normalization["mean"], np.float32)).to(device)
    std = torch.from_numpy(np.asarray(normalization["std"], np.float32)).to(device)
    prediction = np.empty_like(ar)
    gate_sum = residual_sum = gate_count = 0.0
    for start in range(0, len(names), args.batch_size):
        stop = min(start + args.batch_size, len(names))
        contexts, histories, futures = [], [], []
        for name in names[start:stop]:
            with np.load(Path(args.windows) / name, allow_pickle=False) as window:
                contexts.append(window["context_frames"])
                histories.append(window["history_actions"])
                futures.append(window["future_actions"])
        context = torch.from_numpy(np.stack(contexts)).permute(0, 1, 4, 2, 3).to(device).float().div(255.0)
        ar_batch = torch.from_numpy(ar[start:stop].copy()).permute(0, 1, 4, 2, 3).to(device).float().div(255.0)
        actions = torch.from_numpy(np.concatenate((np.stack(histories), np.stack(futures)), axis=1)).to(device).float()
        actions = (actions - mean) / std
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            value, _, _, gate, residual, _ = model(context, ar_batch, actions, return_components=True)
        prediction[start:stop] = value.clamp(0, 1).mul(255).round().byte().permute(0, 1, 3, 4, 2).cpu().numpy()
        gate_sum += float(gate.float().sum())
        residual_sum += float(residual.float().abs().sum())
        gate_count += gate.numel()
        if stop % 100 == 0 or stop == len(names):
            print(json.dumps({"completed": stop, "total": len(names)}), flush=True)
    output_cache = Path(args.output_cache)
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_cache.with_suffix(output_cache.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, prediction=prediction, windows=np.asarray(names))
    os.replace(temporary, output_cache)
    result = {"format": "track2-autoregressive-texture-reprojection-prediction-v1", "checkpoint_dir": str(checkpoint.resolve()), "sample_count": len(names), "gate_mean": gate_sum / gate_count, "residual_abs_mean": residual_sum / (gate_count * 3), "output_cache": str(output_cache.resolve())}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
