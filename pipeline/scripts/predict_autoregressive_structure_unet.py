#!/usr/bin/env python3
"""Roll out a structure-aware autoregressive parent on a reference cache's windows."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from wam_pipeline.autoregressive_structure_unet import OneStepActionSeparatedStructureUNet, OneStepActionStructureUNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--reference-cache", required=True, help="Prediction cache whose ordered window names are reused.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-cache", required=True)
    parser.add_argument("--output-mask-cache", help="Optional uint8 structure-probability cache aligned with output-cache.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with np.load(args.reference_cache, allow_pickle=False) as reference:
        names = [str(value) for value in reference["windows"]]
    checkpoint = Path(args.checkpoint_dir)
    config = np.load(checkpoint / "track2_autoregressive_structure_unet_config.npz", allow_pickle=False)
    if (int(config["context_frames"]), int(config["prediction_frames"]), int(config["action_dim"])) != (5, 8, 14):
        raise ValueError("structure checkpoint does not match Track 2")
    state = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True)
    if state.get("format") not in {"track2-autoregressive-structure-unet-v1", "track2-autoregressive-structure-unet-v2", "track2-autoregressive-structure-unet-v3"}:
        raise ValueError("unsupported structure checkpoint")
    device = torch.device(args.device)
    model_class = OneStepActionSeparatedStructureUNet if state.get("format") == "track2-autoregressive-structure-unet-v3" else OneStepActionStructureUNet
    model = model_class().to(device).eval()
    incompatible = model.load_state_dict(state["state_dict"], strict=False)
    if incompatible.unexpected_keys or set(incompatible.missing_keys) - {"structure_strength"}:
        raise ValueError("structure checkpoint state does not match model")
    normalization = np.load(checkpoint / "action_normalization.npz", allow_pickle=False)
    mean = torch.from_numpy(np.asarray(normalization["mean"], np.float32)).to(device)
    std = torch.from_numpy(np.asarray(normalization["std"], np.float32)).to(device)

    predictions = np.empty((len(names), 8, 256, 256, 3), dtype=np.uint8)
    masks = np.empty((len(names), 8, 256, 256), dtype=np.uint8) if args.output_mask_cache else None
    mask_probability_sum = mask_probability_count = 0.0
    for start in range(0, len(names), args.batch_size):
        stop = min(start + args.batch_size, len(names))
        contexts, histories, futures = [], [], []
        for name in names[start:stop]:
            with np.load(Path(args.windows) / name, allow_pickle=False) as window:
                contexts.append(window["context_frames"])
                histories.append(window["history_actions"])
                futures.append(window["future_actions"])
        context = torch.from_numpy(np.stack(contexts)).permute(0, 1, 4, 2, 3).to(device).float().div(255.0)
        history = (torch.from_numpy(np.stack(histories)).to(device).float() - mean) / std
        future = (torch.from_numpy(np.stack(futures)).to(device).float() - mean) / std
        values, mask_values = [], []
        # Match the serving runtime exactly. Autoregressive bfloat16 roundoff
        # is large enough to accumulate across eight recurrent predictions.
        with torch.inference_mode():
            for action in future.unbind(dim=1):
                value, mask_logit = model(context, torch.cat((history, action[:, None]), dim=1), return_structure=True)
                value = value.clamp(0, 1)
                values.append(value)
                mask_values.append(mask_logit.sigmoid())
                mask_probability_sum += float(mask_logit.float().sigmoid().sum())
                mask_probability_count += mask_logit.numel()
                recurrent = (value + model.structure_correction(mask_logit)).clamp(0, 1)
                context = torch.cat((context[:, 1:], recurrent[:, None]), dim=1)
                history = torch.cat((history[:, 1:], action[:, None]), dim=1)
        rollout = torch.stack(values, dim=1)
        predictions[start:stop] = rollout.mul(255).round().byte().permute(0, 1, 3, 4, 2).cpu().numpy()
        if masks is not None:
            masks[start:stop] = torch.stack(mask_values, dim=1).mul(255).round().byte().squeeze(2).cpu().numpy()
        if stop % 100 == 0 or stop == len(names):
            print(json.dumps({"event": "structure_prediction_progress", "completed": stop, "total": len(names)}), flush=True)

    output_cache = Path(args.output_cache)
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    temporary_cache = output_cache.with_suffix(output_cache.suffix + f".tmp.{os.getpid()}")
    with temporary_cache.open("wb") as handle:
        np.savez_compressed(handle, prediction=predictions, windows=np.asarray(names))
    os.replace(temporary_cache, output_cache)
    output_mask_cache = None
    if masks is not None:
        output_mask_cache = Path(args.output_mask_cache)
        output_mask_cache.parent.mkdir(parents=True, exist_ok=True)
        temporary_mask = output_mask_cache.with_suffix(output_mask_cache.suffix + f".tmp.{os.getpid()}")
        with temporary_mask.open("wb") as handle:
            np.savez_compressed(handle, structure_mask=masks, windows=np.asarray(names))
        os.replace(temporary_mask, output_mask_cache)
    result = {
        "format": "track2-autoregressive-structure-unet-prediction-v1",
        "checkpoint_dir": str(checkpoint.resolve()),
        "reference_cache": str(Path(args.reference_cache).resolve()),
        "sample_count": len(names),
        "mask_probability_mean": mask_probability_sum / mask_probability_count,
        "output_cache": str(output_cache.resolve()),
        "output_mask_cache": str(output_mask_cache.resolve()) if output_mask_cache is not None else None,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
