#!/usr/bin/env python3
"""Generate and score one native-geometry IRASim-v2 Track 2 rollout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, PNDMScheduler

from wam_pipeline.irasim_lora import inject_irasim_lora
from scripts.train_track2_irasim_v2 import load_pretrained_native


def highpass(value: torch.Tensor) -> torch.Tensor:
    flat = value.flatten(0, 1)
    return value - F.avg_pool2d(flat, 3, stride=1, padding=1).view_as(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--irasim-root", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--window", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--actions", help="Optional production-mapped action cache overriding cache/actions")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-mode", choices=("correct", "zero", "reverse", "flip-arm"), default="correct")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.irasim_root).resolve()))
    from models.irasim import IRASim_models
    from sample.pipeline_trajectory2videogen import Trajectory2VideoGenPipeline

    adapter = torch.load(args.adapter, map_location="cpu", weights_only=False)
    config = SimpleNamespace(dataset="bridge", state_dim=8, final_frame_ada=False, gradient_checkpointing=False)
    model = IRASim_models["IRASim-XL/2"](
        input_size=(32, 40), num_frames=13, learn_sigma=False, extras=3, attention_mode="sdpa", args=config
    )
    load_pretrained_native(model, Path(args.pretrained))
    inject_irasim_lora(model, int(adapter["config"]["lora_rank"]), float(adapter["config"]["lora_alpha"]))
    incompatible = model.load_state_dict(adapter["state_dict"], strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"unexpected adapter keys: {incompatible.unexpected_keys[:5]}")
    device = torch.device(args.device)
    model.to(device).eval()
    vae = AutoencoderKL.from_pretrained(args.vae).to(device).eval()
    scheduler = PNDMScheduler.from_pretrained(
        Path(args.irasim_root) / "pretrained_models/scheduler",
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        variance_type="fixed_small",
    )
    pipeline = Trajectory2VideoGenPipeline(vae=vae, scheduler=scheduler, transformer=model)

    stem = Path(args.window).stem
    latent = np.load(Path(args.cache) / "latents" / f"{stem}.npy").astype(np.float32)
    action_root = Path(args.actions) if args.actions else Path(args.cache) / "actions"
    action = np.load(action_root / f"{stem}.npy").astype(np.float32)
    if args.action_mode == "zero":
        action = np.zeros_like(action)
    elif args.action_mode == "reverse":
        action = action[::-1].copy()
    elif args.action_mode == "flip-arm":
        action[:, 7] *= -1
    with np.load(args.window, allow_pickle=False) as item:
        target_u8 = item["target_frames"].copy()

    generator = torch.Generator(device=device).manual_seed(args.seed)
    context = torch.from_numpy(latent[:5]).unsqueeze(0).to(device)
    action_tensor = torch.from_numpy(action).unsqueeze(0).to(device)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        generated, _ = pipeline(
            action_tensor,
            mask_x=context,
            video_length=13,
            height=256,
            width=320,
            num_inference_steps=args.steps,
            guidance_scale=1.0,
            generator=generator,
            device=device,
            output_type="video",
        )
        future = generated[:, 5:].flatten(0, 1) / vae.config.scaling_factor
        decoded = []
        for offset in range(0, len(future), 4):
            decoded.append(vae.decode(future[offset : offset + 4]).sample)
        prediction = torch.cat(decoded).float().add(1).mul(0.5).clamp(0, 1)
        prediction = F.interpolate(prediction, size=(256, 256), mode="bilinear", align_corners=False)
    target = torch.from_numpy(target_u8).permute(0, 3, 1, 2).float().div(255)
    error = (prediction.cpu() - target).abs()
    dark = target.mean(1, keepdim=True) < 0.28
    metrics = {
        "rgb_mae_0_255": float(error.mean() * 255),
        "highpass_mae_0_255": float((highpass(prediction.cpu()) - highpass(target)).abs().mean() * 255),
        "dark_structure_mae_0_255": float(error.mean(1, keepdim=True)[dark].mean() * 255),
    }
    prediction_u8 = prediction.mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output.with_suffix(".npy"), prediction_u8)
    imageio.mimsave(output, [np.concatenate((p, t), axis=1) for p, t in zip(prediction_u8, target_u8)], fps=3, loop=0)
    report = {
        "format": "track2-irasim-v2-single-window-v1",
        "window": str(Path(args.window).resolve()),
        "adapter": str(Path(args.adapter).resolve()),
        "action_mode": args.action_mode,
        "seed": args.seed,
        "steps": args.steps,
        "metrics": metrics,
        "output": str(output.resolve()),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
