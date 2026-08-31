#!/usr/bin/env python3
"""Exercise the real Wan2.2 Track 2 path before starting long training."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as functional

from diffusers import AutoencoderKLWan, WanTransformer3DModel

from train_track2_wan import Track2WindowTorchDataset, frames_to_wan_video
from wam_pipeline.track2_wan import (
    TRACK2_CONTEXT_LATENT_FRAMES,
    TRACK2_LATENT_FRAMES,
    Track2ActionConditioner,
    Track2WanWorldModel,
    build_action_slots,
    denormalize_vae_latents,
    extract_lora_state,
    load_lora_state,
    normalize_vae_latents,
    vae_latent_stats,
    configure_wan_lora,
)


FORMAT = "track2-wan-real-smoke-v1"


def gpu_memory_gb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return round(torch.cuda.max_memory_allocated(device) / 2**30, 3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a real-data Wan2.2 TI2V Track 2 smoke test.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument(
        "--trajectory-conditioned",
        action="store_true",
        help="Exercise the v2 delta-and-relative-pose action residual branch.",
    )
    parser.add_argument(
        "--latent-action-conditioned",
        action="store_true",
        help="Exercise the v3 action residual aligned to Wan VAE temporal slots.",
    )
    args = parser.parse_args()
    if not args.trajectory_conditioned and not args.latent_action_conditioned:
        conditioning_label = "v1"
    elif args.trajectory_conditioned and args.latent_action_conditioned:
        conditioning_label = "v4"
    elif args.trajectory_conditioned:
        conditioning_label = "v2"
    else:
        conditioning_label = "v3"
    if args.lora_rank < 1:
        raise SystemExit("--lora-rank must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    root, base, output = Path(args.dataset_root), Path(args.base_model), Path(args.output)
    if not (base / "transformer").is_dir() or not (base / "vae").is_dir():
        raise SystemExit("--base-model needs complete transformer/ and vae/ directories")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    torch.manual_seed(0)

    dataset = Track2WindowTorchDataset(root, "train")
    if not 0 <= args.window_index < len(dataset):
        raise SystemExit(f"--window-index must be in [0,{len(dataset) - 1}]")
    context, history, future, target = dataset[args.window_index]
    video = frames_to_wan_video(context.unsqueeze(0), target.unsqueeze(0)).to(device)

    # Run VAE independently so the full DiT never shares memory with it during backward.
    vae = AutoencoderKLWan.from_pretrained(
        str(base), subfolder="vae", torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    latent_mean, inverse_std = vae_latent_stats(vae, device=device, dtype=dtype)
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
        raw_latents = vae.encode(video.to(dtype=dtype)).latent_dist.mode()
        clean = normalize_vae_latents(raw_latents, latent_mean, inverse_std)
        reconstructed = vae.decode(raw_latents, return_dict=False)[0]
    if clean.shape != (1, int(vae.config.z_dim), TRACK2_LATENT_FRAMES, 16, 16):
        raise RuntimeError(f"unexpected Wan2.2 latent shape: {tuple(clean.shape)}")
    reconstruction_mae = float((reconstructed.float() - video.float()).abs().mean().cpu())
    clean_cpu = clean.float().cpu()
    del reconstructed, raw_latents, vae
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    transformer = WanTransformer3DModel.from_pretrained(
        str(base), subfolder="transformer", torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    if (int(transformer.config.in_channels), int(transformer.config.out_channels)) != (48, 48):
        raise RuntimeError("official Wan2.2 TI2V-5B must expose 48 input/output channels")
    lora_names = configure_wan_lora(transformer, rank=args.lora_rank)
    conditioner = Track2ActionConditioner(
        int(transformer.config.text_dim),
        trajectory_conditioned=args.trajectory_conditioned,
        latent_action_conditioned=args.latent_action_conditioned,
        latent_channels=int(transformer.config.out_channels) if args.latent_action_conditioned else None,
    ).to(device, dtype=dtype)
    model = Track2WanWorldModel(transformer, conditioner).to(device).train()
    model.transformer.enable_gradient_checkpointing()
    clean = clean_cpu.to(device=device, dtype=dtype)
    slots = build_action_slots(history.unsqueeze(0).to(device), future.unsqueeze(0).to(device))
    noise = torch.randn_like(clean)
    sigma = torch.tensor([0.5], device=device, dtype=dtype)
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
        velocity, flow_target, mixed = model(clean, noise, sigma, slots)
        loss = functional.mse_loss(
            velocity[:, :, TRACK2_CONTEXT_LATENT_FRAMES:].float(),
            flow_target[:, :, TRACK2_CONTEXT_LATENT_FRAMES:].float(),
        )
    loss.backward()
    gradient_l1 = sum(
        float(parameter.grad.detach().abs().sum().cpu())
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    )
    if not torch.isfinite(loss) or gradient_l1 <= 0:
        raise RuntimeError("real Wan forward/backward produced invalid or disconnected gradients")
    dit_input_shape = list(mixed.shape)
    token_timestep = model._timestep_for_model(mixed, sigma * 1000.0)
    expected_context_tokens = TRACK2_CONTEXT_LATENT_FRAMES * (16 // 2) * (16 // 2)
    if token_timestep.shape != (1, TRACK2_LATENT_FRAMES * (16 // 2) * (16 // 2)):
        raise RuntimeError(f"unexpected Wan2.2 token timestep shape: {tuple(token_timestep.shape)}")
    if not torch.equal(token_timestep[:, :expected_context_tokens], torch.zeros_like(token_timestep[:, :expected_context_tokens])):
        raise RuntimeError("Wan2.2 context latent tokens were not assigned timestep zero")
    if not torch.equal(
        token_timestep[:, expected_context_tokens:],
        torch.full_like(token_timestep[:, expected_context_tokens:], 500.0),
    ):
        raise RuntimeError("Wan2.2 future latent tokens were not assigned the sampled timestep")
    trainable_tensor_count = len(lora_names) + len(list(conditioner.parameters()))
    lora_state = extract_lora_state(model.transformer, "track2")
    checkpoint = {
        "format": FORMAT,
        "lora_rank": args.lora_rank,
        "action_conditioner": {name: value.detach().cpu() for name, value in model.action_conditioner.state_dict().items()},
        "transformer_lora": lora_state,
    }
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "real_smoke_lora.pt"
    torch.save(checkpoint, checkpoint_path)
    reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if set(reloaded["transformer_lora"]) != set(lora_state):
        raise RuntimeError("saved real Wan LoRA checkpoint did not round-trip")
    transformer_memory_gb = gpu_memory_gb(device)
    del model, transformer, conditioner, mixed, velocity, flow_target, noise
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # Match the inference process: attach a fresh adapter and load the compact
    # checkpoint before claiming the saved state is deployable.
    reloaded_transformer = WanTransformer3DModel.from_pretrained(
        str(base), subfolder="transformer", torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    configure_wan_lora(reloaded_transformer, rank=args.lora_rank)
    reloaded_conditioner = Track2ActionConditioner(
        int(reloaded_transformer.config.text_dim),
        trajectory_conditioned=args.trajectory_conditioned,
        latent_action_conditioned=args.latent_action_conditioned,
        latent_channels=int(reloaded_transformer.config.out_channels) if args.latent_action_conditioned else None,
    ).to(device, dtype=dtype)
    reloaded_conditioner.load_state_dict(reloaded["action_conditioner"], strict=True)
    load_lora_state(reloaded_transformer, "track2", reloaded["transformer_lora"])
    del reloaded_transformer, reloaded_conditioner
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # Decode from the previously encoded official window after the DiT is released.
    vae = AutoencoderKLWan.from_pretrained(
        str(base), subfolder="vae", torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()
    decode_mean, decode_inverse_std = vae_latent_stats(vae, device=device, dtype=dtype)
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
        decoded = vae.decode(
            denormalize_vae_latents(clean_cpu.to(device=device, dtype=dtype), decode_mean, decode_inverse_std),
            return_dict=False,
        )[0]
    if decoded.shape != (1, 3, 13, 256, 256):
        raise RuntimeError(f"unexpected decoded Wan2.2 video shape: {tuple(decoded.shape)}")
    report = {
        "format": FORMAT,
        "status": "passed",
        "dataset_root": str(root.resolve()),
        "base_model": str(base.resolve()),
        "window_index": args.window_index,
        "raw_video_shape": list(video.shape),
        "latent_shape": list(clean_cpu.shape),
        "dit_input_shape": dit_input_shape,
        "token_timestep_shape": list(token_timestep.shape),
        "context_token_timestep": 0.0,
        "future_token_timestep": 500.0,
        "decoded_shape": list(decoded.shape),
        "vae_reconstruction_mae_minus1_to1": reconstruction_mae,
        "future_flow_mse": float(loss.detach().cpu()),
        "trajectory_conditioned": args.trajectory_conditioned,
        "latent_action_conditioned": args.latent_action_conditioned,
        "conditioning_label": conditioning_label,
        "trainable_tensor_count": trainable_tensor_count,
        "gradient_l1": gradient_l1,
        "transformer_backward_peak_gpu_gb": transformer_memory_gb,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_load_roundtrip": True,
    }
    (output / "real_smoke_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
