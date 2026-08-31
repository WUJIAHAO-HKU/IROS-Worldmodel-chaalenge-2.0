#!/usr/bin/env python3
"""Run the local model-based RL smoke test against one prepared window."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from wam_pipeline.backends import build_backend
from wam_pipeline.data import load_window_npz
from wam_pipeline.mb_rl import run_smoke


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", required=True)
    parser.add_argument(
        "--backend",
        choices=("synthetic", "ivideogpt", "track2-wan", "official-rlinf-wan", "residual-unet", "flow-residual-unet", "temporal-unet", "direct-video-unet", "direct-flow-unet", "wan-flow-ensemble", "autoregressive-flow-ensemble", "local-motion-texture-fusion", "multisource-flow-unet", "autoregressive-unet", "hybrid-unet"),
        default="synthetic",
    )
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--strict-evaluation", help="Required for every non-synthetic world-model MBRL run.")
    parser.add_argument("--windows", default="artifacts/adjust_bottle_windows_full")
    parser.add_argument("--split-manifest", default="artifacts/splits/adjust_bottle_50episodes_full.json")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wan-base-model")
    parser.add_argument("--wan-inference-steps", type=int, default=30)
    parser.add_argument("--wan-inference-solver", choices=("euler", "heun"), default="euler")
    parser.add_argument("--official-diffsynth-root")
    parser.add_argument("--official-wan-inference-steps", type=int, default=5)
    args = parser.parse_args()
    if args.backend != "synthetic":
        if not args.strict_evaluation or not args.checkpoint_dir:
            raise SystemExit("non-synthetic MBRL requires --checkpoint-dir and --strict-evaluation from the full validation pass")
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().with_name("require_strict_world_model_acceptance.py")),
                "--evaluation", args.strict_evaluation,
                "--windows", args.windows,
                "--split-manifest", args.split_manifest,
                "--checkpoint-dir", args.checkpoint_dir,
                "--backend", args.backend,
                *(["--wan-base-model", args.wan_base_model] if args.wan_base_model else []),
            ],
            check=True,
        )
    result = run_smoke(
        load_window_npz(args.window),
        build_backend(
            args.backend,
            args.checkpoint_dir,
            args.device,
            wan_base_model=args.wan_base_model,
            wan_inference_steps=args.wan_inference_steps,
            wan_inference_solver=args.wan_inference_solver,
            official_diffsynth_root=args.official_diffsynth_root,
            official_wan_inference_steps=args.official_wan_inference_steps,
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
