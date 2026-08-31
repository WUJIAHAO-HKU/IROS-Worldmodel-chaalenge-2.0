#!/usr/bin/env python3
"""Verify two non-overlapping 8-step closed-loop calls to a Track 2 backend."""

from __future__ import annotations

import argparse
import json

import numpy as np

from wam_pipeline.backends import build_backend
from wam_pipeline.data import load_window_npz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", required=True)
    parser.add_argument(
        "--backend",
        choices=("synthetic", "ivideogpt", "track2-wan", "official-rlinf-wan", "residual-unet", "flow-residual-unet", "temporal-unet", "direct-video-unet", "direct-flow-unet", "wan-flow-ensemble", "autoregressive-flow-ensemble", "local-motion-texture-fusion", "multisource-flow-unet", "autoregressive-unet", "hybrid-unet"),
        required=True,
    )
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wan-base-model")
    parser.add_argument("--wan-inference-steps", type=int, default=30)
    parser.add_argument("--wan-inference-solver", choices=("euler", "heun"), default="euler")
    parser.add_argument("--official-diffsynth-root")
    parser.add_argument("--official-wan-inference-steps", type=int, default=5)
    args = parser.parse_args()
    window = load_window_npz(args.window)
    backend = build_backend(
        args.backend,
        args.checkpoint_dir,
        args.device,
        wan_base_model=args.wan_base_model,
        wan_inference_steps=args.wan_inference_steps,
        wan_inference_solver=args.wan_inference_solver,
        official_diffsynth_root=args.official_diffsynth_root,
        official_wan_inference_steps=args.official_wan_inference_steps,
    )
    first_actions = window.future_actions.copy()
    first = backend.predict(window.context_frames, window.history_actions, first_actions, seed=0, instruction=None)
    # Eight transitions have elapsed: p3..p7 are five observations, and u4..u7
    # are the four actions that produced the last four observations.
    second_context = first[-5:]
    second_history = first_actions[-4:]
    second_actions = first_actions.copy()
    second = backend.predict(second_context, second_history, second_actions, seed=1, instruction=None)
    print(
        json.dumps(
            {
                "rounds": 2,
                "frames_per_round": 8,
                "first_shape": list(first.shape),
                "second_shape": list(second.shape),
                "next_context_shape": list(second_context.shape),
                "next_history_actions_shape": list(second_history.shape),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
