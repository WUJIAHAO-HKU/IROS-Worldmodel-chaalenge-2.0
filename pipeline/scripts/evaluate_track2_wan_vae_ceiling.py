#!/usr/bin/env python3
"""Measure the RGB error introduced by the frozen Wan VAE alone.

This is a feasibility diagnostic, not a world-model evaluation: it encodes and
decodes the true 13-frame video, then measures its eight future RGB frames.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from diffusers import AutoencoderKLWan

from scripts.train_track2_wan import Track2WindowTorchDataset, frames_to_wan_video
from wam_pipeline.profile import CONTEXT_FRAMES
from wam_pipeline.track2_wan import denormalize_vae_latents, normalize_vae_latents, vae_latent_stats


def selected_indices(total: int, samples: int) -> np.ndarray:
    if samples < 0:
        raise ValueError("--samples must be non-negative")
    if samples == 0 or samples >= total:
        return np.arange(total, dtype=np.int64)
    return np.linspace(0, total - 1, num=samples, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frozen Wan VAE RGB reconstruction ceiling.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--samples", type=int, default=0, help="0 evaluates the full selected split.")
    parser.add_argument("--accept-mae", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.accept_mae <= 0:
        raise SystemExit("--accept-mae must be positive")

    root, base = Path(args.dataset_root), Path(args.base_model)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA evaluation requested but CUDA is unavailable")
    dataset = Track2WindowTorchDataset(root, args.split)
    indices = selected_indices(len(dataset), args.samples)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    vae = AutoencoderKLWan.from_pretrained(
        str(base), subfolder="vae", torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()
    latent_mean, latent_std = vae_latent_stats(vae, device=device, dtype=dtype)

    window_rows: list[dict] = []
    frame_maes: list[np.ndarray] = []
    for index in indices:
        context, _, _, target = dataset[int(index)]
        video = frames_to_wan_video(context.unsqueeze(0), target.unsqueeze(0)).to(device, dtype=dtype)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=dtype, enabled=device.type == "cuda"
        ):
            raw = vae.encode(video).latent_dist.mode()
            # Exercise the same normalized latent round-trip used by runtime inference.
            decoded = vae.decode(
                denormalize_vae_latents(
                    normalize_vae_latents(raw, latent_mean, latent_std), latent_mean, latent_std
                ),
                return_dict=False,
            )[0]
        prediction = decoded[:, :, CONTEXT_FRAMES:].float().clamp(-1, 1).add(1).mul(127.5)
        truth = video[:, :, CONTEXT_FRAMES:].float().clamp(-1, 1).add(1).mul(127.5)
        per_frame = (prediction - truth).abs().mean(dim=(1, 3, 4))[0].cpu().numpy()
        frame_maes.append(per_frame)
        window_rows.append(
            {
                "dataset_index": int(index),
                "mean_mae_over_8_frames": float(per_frame.mean()),
                "max_prediction_frame_mae": float(per_frame.max()),
                "mae_by_prediction_frame": [float(value) for value in per_frame],
                "passes_all_prediction_frames": bool((per_frame < args.accept_mae).all()),
            }
        )

    values = np.asarray(frame_maes, dtype=np.float32)
    result = {
        "format": "track2-wan-vae-ceiling-v1",
        "kind": "frozen_vae_reconstruction_diagnostic_not_world_model_prediction",
        "dataset_root": str(root.resolve()),
        "base_model": str(base.resolve()),
        "split": args.split,
        "sample_count": int(len(indices)),
        "threshold_mae_0_255": float(args.accept_mae),
        "vae_mae_by_prediction_frame": [float(value) for value in values.mean(axis=0)],
        "vae_mae_mean": float(values.mean()),
        "max_window_mean_over_8_frames": float(values.mean(axis=1).max()),
        "max_prediction_frame_mae": float(values.max()),
        "windows_passing_all_8_prediction_frames": int((values.max(axis=1) < args.accept_mae).sum()),
        "all_windows_and_frames_pass": bool((values < args.accept_mae).all()),
        "window_reconstruction": window_rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
