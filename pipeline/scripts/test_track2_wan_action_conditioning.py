#!/usr/bin/env python3
"""CPU/GPU smoke test for the 14-D Track 2 Wan action path.

This test creates a deliberately tiny Wan DiT.  It verifies the exact 13-slot
contract, future-action output sensitivity, non-zero future-action gradients,
and that observed latent slots remain immutable through sampling.  It does not
claim a tiny randomly initialized DiT is a trained world model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from diffusers import WanTransformer3DModel

from wam_pipeline.track2_wan import (
    TRACK2_CONTEXT_LATENT_FRAMES,
    TRACK2_LATENT_FRAMES,
    Track2ActionConditioner,
    Track2WanWorldModel,
    build_action_slots,
)


def tiny_transformer() -> WanTransformer3DModel:
    return WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=4,
        out_channels=4,
        text_dim=32,
        freq_dim=16,
        ffn_dim=64,
        num_layers=2,
        cross_attn_norm=True,
        rope_max_seq_len=16,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report")
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
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    torch.manual_seed(args.seed)

    transformer = tiny_transformer()
    conditioner = Track2ActionConditioner(
        32,
        hidden_dim=32,
        trajectory_conditioned=args.trajectory_conditioned,
        latent_action_conditioned=args.latent_action_conditioned,
        latent_channels=int(transformer.config.out_channels) if args.latent_action_conditioned else None,
    )
    model = Track2WanWorldModel(transformer, conditioner).to(device)
    clean = torch.randn((2, 4, TRACK2_LATENT_FRAMES, 4, 4), device=device)
    noise = torch.randn_like(clean)
    history = torch.randn((2, 4, 14), device=device)
    future = torch.randn((2, 8, 14), device=device, requires_grad=True)
    slots = build_action_slots(history, future)
    velocity, target, mixed = model(clean, noise, torch.tensor((0.2, 0.8), device=device), slots)
    loss = (velocity[:, :, TRACK2_CONTEXT_LATENT_FRAMES:] - target[:, :, TRACK2_CONTEXT_LATENT_FRAMES:]).square().mean()
    loss.backward()
    gradient_l1 = float(future.grad.abs().sum().detach().cpu())

    perturbed = slots.detach().clone()
    perturbed[:, 5:, 0] += 1.0
    with torch.no_grad():
        altered = model.predict_velocity(mixed, clean, torch.tensor((200.0, 800.0), device=device), perturbed)
        delta = float((velocity.detach() - altered).abs().mean().cpu())
        generator = torch.Generator(device=device).manual_seed(args.seed)
        sampled = model.sample_future_latents(clean, slots.detach(), num_inference_steps=2, generator=generator)
        context_error = float((sampled[:, :, :TRACK2_CONTEXT_LATENT_FRAMES] - clean[:, :, :TRACK2_CONTEXT_LATENT_FRAMES]).abs().max().cpu())
        latent_alignment_error = None
        if args.latent_action_conditioned:
            features = conditioner.latent_action_features(slots.detach())
            if tuple(features.shape) != (2, TRACK2_LATENT_FRAMES, 42):
                raise SystemExit(f"unexpected latent action feature shape: {tuple(features.shape)}")
            changed = slots.detach().clone()
            changed[:, 5:9, 0] += 1.0
            changed_features = conditioner.latent_action_features(changed)
            # Changing raw frames 5--8 must affect only Wan latent slot two.
            difference = (changed_features - features).abs().sum(dim=-1)
            latent_alignment_error = float(torch.cat((difference[:, :2], difference[:, 3:]), dim=1).max().cpu())

    if gradient_l1 <= 0:
        raise SystemExit("future actions have zero gradient; action conditioning is disconnected")
    if delta <= 0:
        raise SystemExit("future action perturbation did not alter Wan velocity")
    if context_error != 0:
        raise SystemExit("Wan sampler modified observed context latent slots")
    if latent_alignment_error is not None and latent_alignment_error != 0:
        raise SystemExit("a future action group leaked into the wrong Wan latent slot")
    result = {
        "format": (
            "track2-wan-action-conditioning-test-v4"
            if args.trajectory_conditioned and args.latent_action_conditioned
            else "track2-wan-action-conditioning-test-v3"
            if args.latent_action_conditioned
            else "track2-wan-action-conditioning-test-v2"
            if args.trajectory_conditioned
            else "track2-wan-action-conditioning-test-v1"
        ),
        "status": "passed",
        "device": str(device),
        "action_slots": int(slots.shape[1]),
        "action_dim": int(slots.shape[2]),
        "latent_frames": TRACK2_LATENT_FRAMES,
        "context_latent_frames": TRACK2_CONTEXT_LATENT_FRAMES,
        "future_action_gradient_l1": gradient_l1,
        "future_action_perturbation_velocity_mae": delta,
        "trajectory_conditioned": args.trajectory_conditioned,
        "latent_action_conditioned": args.latent_action_conditioned,
        "latent_action_cross_slot_error": latent_alignment_error,
        "observed_context_sample_max_abs_error": context_error,
    }
    if args.report:
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
