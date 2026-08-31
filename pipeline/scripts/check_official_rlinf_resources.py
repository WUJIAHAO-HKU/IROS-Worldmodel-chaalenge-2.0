#!/usr/bin/env python3
"""Fail fast unless required public components for a local RLinf run exist."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow direct ``python pipeline/scripts/...`` execution as documented.
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from wam_pipeline.official_rlinf_wan import validate_official_rlinf_wan_checkpoint


def require_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing {label}: {path}")


def require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise SystemExit(f"missing {label}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resources", default="artifacts/official_resources")
    parser.add_argument("--rlinf-root", default="third_party/WorldArena-2.0/RL_env_benchmark")
    parser.add_argument("--robotwin", default=None)
    parser.add_argument("--wan", default=None, help="Optional DiffSynth source for Wan-only recipes")
    parser.add_argument(
        "--wan-checkpoint",
        default=None,
        help="Optional published Wan checkpoint; not used by the Track-2 HTTP bridge",
    )
    parser.add_argument(
        "--reset-dataset",
        default=None,
        help="Required by the HTTP proxy environment; may be generated from public HDF5 data",
    )
    args = parser.parse_args()
    resources = Path(args.resources)
    policy = resources / "pi05_adjust_bottle"
    reward = resources / "reward_model"
    require_file(policy / "model.safetensors", "official pi05 model.safetensors")
    require_file(policy / "metadata.pt", "official pi05 metadata.pt")
    require_file(
        policy / "rlinf/robotwin_headcam_adjust_bottle/norm_stats.json", "official pi05 RLinf norm statistics"
    )
    require_file(reward / "adjust_bottle/full_weights.pt", "official Adjust Bottle reward checkpoint")
    for name in ("config.json", "spiece.model", "model.safetensors"):
        require_file(reward / "t5-base" / name, f"local t5-base {name}")
    require_directory(Path(args.rlinf_root) / "rlinf", "official RLinf source")
    if args.robotwin:
        robotwin = Path(args.robotwin)
        require_directory(robotwin / "envs", "RoboTwin RLinf_support source")
        require_directory(robotwin / "assets", "downloaded RoboTwin assets")
    if args.wan:
        require_directory(Path(args.wan) / "diffsynth", "DiffSynth Studio source")
    if args.wan_checkpoint:
        validate_official_rlinf_wan_checkpoint(args.wan_checkpoint)
    if args.reset_dataset:
        reset_dataset = Path(args.reset_dataset)
        require_directory(reset_dataset, "RLinf reset dataset")
        if not any(reset_dataset.glob("*.npy")):
            raise SystemExit(f"reset dataset has no .npy trajectory files: {reset_dataset}")
    print("official RLinf resource check passed")


if __name__ == "__main__":
    main()
