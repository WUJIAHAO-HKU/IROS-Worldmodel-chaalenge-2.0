#!/usr/bin/env python3
"""Run the local model-based RL smoke test against one prepared window."""

from __future__ import annotations

import argparse
import json
import os

from wam_pipeline.backends import build_backend
from wam_pipeline.data import load_window_npz
from wam_pipeline.mb_rl import run_smoke


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", required=True)
    parser.add_argument(
        "--backend",
        choices=("synthetic", "ivideogpt", "residual-unet", "v15-composite"),
        default="synthetic",
    )
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--v15-library-dir", default=os.environ.get("WAM_V15_LIBRARY_DIR"))
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    result = run_smoke(
        load_window_npz(args.window),
        build_backend(
            args.backend,
            args.checkpoint_dir,
            v15_library_dir=args.v15_library_dir,
        ),
        rounds=args.rounds,
    )
    print(json.dumps({
        "rounds": result.rounds,
        "reward": result.reward,
        "generated_frames": result.generated_frames,
        "policy_mean_action_norm": float((result.policy_mean_action**2).sum() ** 0.5),
    }, indent=2))


if __name__ == "__main__":
    main()
