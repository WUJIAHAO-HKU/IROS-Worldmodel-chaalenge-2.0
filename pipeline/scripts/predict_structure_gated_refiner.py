#!/usr/bin/env python3
"""Apply a trained structure-gated high-frequency refiner to aligned caches."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from wam_pipeline.structure_gated_refiner import StructureGatedHighFrequencyRefiner


def prediction_cache(path):
    with np.load(path, allow_pickle=False) as cache:
        return cache["prediction"], [str(value) for value in cache["windows"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--baseline-cache", required=True)
    parser.add_argument("--structure-cache", required=True)
    parser.add_argument("--mask-cache", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    baseline, names = prediction_cache(args.baseline_cache)
    structure, structure_names = prediction_cache(args.structure_cache)
    with np.load(args.mask_cache, allow_pickle=False) as cache:
        masks = cache["structure_mask"]
        mask_names = [str(value) for value in cache["windows"]]
    if names != structure_names or names != mask_names or baseline.shape != structure.shape or masks.shape != baseline.shape[:2] + baseline.shape[2:4]:
        raise ValueError("input caches are not aligned")
    checkpoint = Path(args.checkpoint_dir)
    config = np.load(checkpoint / "structure_refiner_config.npz", allow_pickle=False)
    model = StructureGatedHighFrequencyRefiner(int(config["base_channels"]), float(config["residual_scale"]))
    state = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True)
    if state.get("format") != "track2-structure-gated-high-frequency-refiner-v1":
        raise ValueError("unsupported refiner checkpoint")
    model.load_state_dict(state["state_dict"], strict=True)
    device = torch.device(args.device)
    model = model.to(device).eval()
    normalization = np.load(checkpoint / "action_normalization.npz", allow_pickle=False)
    mean = torch.from_numpy(np.asarray(normalization["mean"], np.float32)).to(device)
    std = torch.from_numpy(np.asarray(normalization["std"], np.float32)).to(device)
    predictions = np.empty_like(baseline)
    gate_sum = support_sum = correction_sum = gate_count = 0.0
    for start in range(0, len(names), args.batch_size):
        stop = min(start + args.batch_size, len(names))
        contexts, actions = [], []
        for name in names[start:stop]:
            with np.load(Path(args.windows) / name, allow_pickle=False) as window:
                contexts.append(window["context_frames"][-1:])
                actions.append(window["future_actions"])
        context = torch.from_numpy(np.stack(contexts)).permute(0, 1, 4, 2, 3).to(device).float().div(255.0)
        baseline_batch = torch.from_numpy(baseline[start:stop].copy()).permute(0, 1, 4, 2, 3).to(device).float().div(255.0)
        structure_batch = torch.from_numpy(structure[start:stop].copy()).permute(0, 1, 4, 2, 3).to(device).float().div(255.0)
        mask_batch = torch.from_numpy(masks[start:stop, :, None].copy()).to(device).float().div(255.0)
        action_batch = (torch.from_numpy(np.stack(actions)).to(device).float() - mean) / std
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction, gate, correction, support = model(context, baseline_batch, structure_batch, mask_batch, action_batch)
        predictions[start:stop] = prediction.clamp(0, 1).mul(255).round().byte().permute(0, 1, 3, 4, 2).cpu().numpy()
        gate_sum += float(gate.float().sum())
        support_sum += float(support.float().sum())
        correction_sum += float(correction.float().abs().sum())
        gate_count += gate.numel()
        if stop % 100 == 0 or stop == len(names):
            print(json.dumps({"event": "structure_refiner_progress", "completed": stop, "total": len(names)}), flush=True)
    output_cache = Path(args.output_cache)
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    temporary_cache = output_cache.with_suffix(output_cache.suffix + f".tmp.{os.getpid()}")
    with temporary_cache.open("wb") as handle:
        np.savez_compressed(handle, prediction=predictions, windows=np.asarray(names))
    os.replace(temporary_cache, output_cache)
    result = {"format": "track2-structure-gated-high-frequency-refiner-prediction-v1", "checkpoint_dir": str(checkpoint.resolve()), "sample_count": len(names), "gate_mean": gate_sum / gate_count, "support_mean": support_sum / gate_count, "correction_abs_mean": correction_sum / (gate_count * 3.0), "output_cache": str(output_cache.resolve())}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
