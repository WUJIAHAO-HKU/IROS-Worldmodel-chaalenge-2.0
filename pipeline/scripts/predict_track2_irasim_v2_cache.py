#!/usr/bin/env python3
"""Batch-cache deterministic native-geometry IRASim-v2 rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, PNDMScheduler

from wam_pipeline.irasim_lora import inject_irasim_lora
from scripts.train_track2_irasim_v2 import load_pretrained_native


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--irasim-root", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--actions", help="Optional production-mapped action cache overriding cache/actions")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
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
    model.load_state_dict(adapter["state_dict"], strict=False)
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

    cache = Path(args.cache)
    all_names = sorted(path.stem for path in (cache / "latents").glob("*.npy"))
    indices = np.linspace(0, len(all_names) - 1, min(args.sample_count, len(all_names)), dtype=int)
    names = [all_names[index] for index in indices]
    predictions = []
    action_root = Path(args.actions) if args.actions else cache / "actions"
    generator = torch.Generator(device=device).manual_seed(args.seed)
    for start in range(0, len(names), args.batch_size):
        current = names[start : start + args.batch_size]
        latent = torch.from_numpy(
            np.stack([np.load(cache / "latents" / f"{name}.npy")[:5] for name in current]).astype(np.float32)
        ).to(device)
        action_value = np.stack([np.load(action_root / f"{name}.npy") for name in current]).astype(np.float32)
        if args.action_mode == "zero":
            action_value.fill(0)
        elif args.action_mode == "reverse":
            action_value = action_value[:, ::-1].copy()
        elif args.action_mode == "flip-arm":
            action_value[:, :, 7] *= -1
        actions = torch.from_numpy(action_value).to(device)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            generated, _ = pipeline(
                actions,
                mask_x=latent,
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
            for offset in range(0, len(future), 8):
                decoded.append(vae.decode(future[offset : offset + 8]).sample)
            rgb = torch.cat(decoded).float().add(1).mul(0.5).clamp(0, 1)
            rgb = F.interpolate(rgb, size=(256, 256), mode="bilinear", align_corners=False)
            rgb = rgb.mul(255).round().byte().unflatten(0, (len(current), 8)).permute(0, 1, 3, 4, 2).cpu().numpy()
        predictions.append(rgb)
        print(json.dumps({"completed": min(start + args.batch_size, len(names)), "total": len(names)}), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    window_names = np.asarray([f"{name}.npz" for name in names])
    np.savez_compressed(output, prediction=np.concatenate(predictions), windows=window_names)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "format": "track2-irasim-v2-cache-v1",
                "adapter": str(Path(args.adapter).resolve()),
                "seed": args.seed,
                "steps": args.steps,
                "action_mode": args.action_mode,
                "sample_count": len(names),
                "windows": window_names.tolist(),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
