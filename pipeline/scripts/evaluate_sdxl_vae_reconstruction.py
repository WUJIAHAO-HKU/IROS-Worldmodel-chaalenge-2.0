#!/usr/bin/env python3
"""Measure the pixel/detail ceiling imposed by the SDXL VAE used by IRASim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL


def highpass(value: torch.Tensor) -> torch.Tensor:
    flat = value.flatten(0, 1)
    return value - F.avg_pool2d(flat, 3, stride=1, padding=1).view_as(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", required=True)
    parser.add_argument("--latent", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with np.load(args.window, allow_pickle=False) as item:
        target_u8 = item["target_frames"]
    latent = torch.from_numpy(np.load(args.latent)[5:].astype(np.float32)).to(args.device)
    vae = AutoencoderKL.from_pretrained(args.vae).to(args.device).eval()
    with torch.inference_mode():
        decoded = vae.decode(latent / vae.config.scaling_factor).sample
    prediction = decoded.add(1).mul(0.5).clamp(0, 1).cpu()
    target = torch.from_numpy(target_u8.copy()).permute(0, 3, 1, 2).float().div(255)
    pixel = (prediction - target).abs()
    dark = target.mean(1, keepdim=True) < 0.28
    edge = (highpass(prediction) - highpass(target)).abs()
    metrics = {
        "rgb_mae_0_255": float(pixel.mean() * 255),
        "highpass_mae_0_255": float(edge.mean() * 255),
        "dark_structure_mae_0_255": float(pixel.mean(1, keepdim=True)[dark].mean() * 255),
        "dark_pixel_fraction": float(dark.float().mean()),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    prediction_u8 = prediction.mul(255).round().byte().permute(0, 2, 3, 1).numpy()
    frames = [np.concatenate((pred, truth), axis=1) for pred, truth in zip(prediction_u8, target_u8)]
    imageio.mimsave(output / "reconstruction_vs_ground_truth.gif", frames, duration=250, loop=0)
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
