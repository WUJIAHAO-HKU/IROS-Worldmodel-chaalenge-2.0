#!/usr/bin/env python3
"""Load the published pi05 Adjust Bottle policy and generate one action chunk."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/official_resources/pi05_adjust_bottle")
    parser.add_argument("--window", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--instruction", default="adjust bottle")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    root = Path(args.checkpoint).resolve()
    required = [root / "model.safetensors", root / "metadata.pt", root / "rlinf/robotwin_headcam_adjust_bottle/norm_stats.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing official policy files: " + ", ".join(missing))

    # RLinf's config intentionally resolves the norm stats from this checkpoint.
    os.environ["OPENPI_CKPT_PATH"] = str(root)
    # The published recipe declares `model_type: openpi`; CFG is a different
    # RLinf model variant and cannot validate this initial policy.
    from rlinf.models.embodiment.openpi import get_model
    from wam_pipeline.data import load_window_npz

    window = load_window_npz(args.window)
    cfg = OmegaConf.create(
        {
            "model_path": str(root),
            "openpi_data": None,
            "openpi": {
                "config_name": "pi05_aloha_robotwin_head_adjust_bottle",
                "num_images_in_input": 1,
                "action_chunk": 8,
                "action_env_dim": 14,
                "num_steps": 5,
                "add_value_head": False,
                "train_expert_only": True,
                "noise_method": "flow_sde",
            },
        }
    )
    # This follows the exact observation emitted by WanEnv._wrap_obs: the
    # latest head-camera image, a zero 14D state, and one task description.
    model = get_model(cfg)
    model = model.to(args.device).eval()
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    frame = torch.from_numpy(window.context_frames[-1:]).to(args.device)
    states = torch.zeros((1, 14), dtype=torch.float32, device=args.device)
    with torch.inference_mode():
        actions, _ = model.predict_action_batch(
            {
                "main_images": frame,
                "wrist_images": None,
                "extra_view_images": None,
                "states": states,
                "task_descriptions": [args.instruction],
            },
            mode="eval",
            compute_values=False,
        )
    if actions.shape != (1, 8, 14) or not torch.isfinite(actions).all():
        raise RuntimeError(f"official pi05 returned invalid action tensor: {tuple(actions.shape)}")
    print(
        json.dumps(
            {
                "checkpoint": str(root),
                "action_shape": list(actions.shape),
                "action_min": float(actions.min().cpu()),
                "action_max": float(actions.max().cpu()),
                "instruction": args.instruction,
                "seed": args.seed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
