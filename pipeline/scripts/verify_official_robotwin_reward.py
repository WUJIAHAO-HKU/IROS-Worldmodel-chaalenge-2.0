#!/usr/bin/env python3
"""Score generated Track 2 frames with the published RoboTwin reward model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", required=True)
    parser.add_argument("--backend", default="residual-unet", choices=("synthetic", "ivideogpt", "residual-unet"))
    parser.add_argument(
        "--backend-checkpoint",
        default="artifacts/checkpoints/residual-unet-track2-native256-5000/best",
    )
    parser.add_argument(
        "--checkpoint",
        default="artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt",
    )
    parser.add_argument("--t5-model", default="artifacts/official_resources/reward_model/t5-base")
    parser.add_argument("--instruction", default="adjust bottle")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    t5_model = Path(args.t5_model)
    required = [checkpoint, *(t5_model / name for name in ("config.json", "spiece.model", "model.safetensors"))]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing official reward files: " + ", ".join(missing))

    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel
    from wam_pipeline.backends import build_backend
    from wam_pipeline.data import load_window_npz

    window = load_window_npz(args.window)
    predictions = build_backend(args.backend, args.backend_checkpoint, args.device).predict(
        window.context_frames,
        window.history_actions,
        window.future_actions,
        seed=0,
        instruction=args.instruction,
    )
    model = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        str(checkpoint), config={"t5_model_name": str(t5_model)}
    ).eval().to(args.device)
    images = torch.from_numpy(predictions).permute(0, 3, 1, 2).float().div(255.0)
    with torch.inference_mode():
        rewards = model.compute_reward(images, [args.instruction] * len(images))
    if rewards.shape != (8,) or not torch.isfinite(rewards).all():
        raise RuntimeError(f"official reward returned invalid tensor: {tuple(rewards.shape)}")
    print(
        json.dumps(
            {
                "reward_checkpoint": str(checkpoint.resolve()),
                "t5_model": str(t5_model.resolve()),
                "input": "8 generated Track 2 RGB frames",
                "reward_shape": list(rewards.shape),
                "reward_min": float(rewards.min().cpu()),
                "reward_max": float(rewards.max().cpu()),
                "reward_mean": float(rewards.mean().cpu()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
