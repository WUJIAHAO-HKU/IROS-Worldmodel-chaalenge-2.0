#!/usr/bin/env python3
"""Package two horizon-specialized checkpoints as one portable Track 2 candidate."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def copy_checkpoint(source: Path, destination: Path, required: tuple[str, ...]) -> None:
    if not source.is_dir():
        raise SystemExit(f"missing checkpoint directory: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    for name in required:
        path = source / name
        if not path.is_file():
            raise SystemExit(f"missing checkpoint file: {path}")
        shutil.copy2(path, destination / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--autoregressive", required=True)
    parser.add_argument("--direct", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--autoregressive-frames", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.autoregressive_frames < 8:
        raise SystemExit("--autoregressive-frames must be in [1, 7]")
    output = Path(args.output)
    copy_checkpoint(
        Path(args.autoregressive),
        output / "autoregressive",
        ("model.pt", "action_normalization.npz", "track2_autoregressive_unet_config.npz", "training_manifest.json"),
    )
    copy_checkpoint(
        Path(args.direct),
        output / "direct",
        ("model.pt", "action_normalization.npz", "track2_residual_unet_config.npz", "training_manifest.json"),
    )
    metadata = {
        "format": "track2-horizon-hybrid-v1",
        "autoregressive_frames": args.autoregressive_frames,
        "prediction_frames": 8,
        "action_dim": 14,
        "autoregressive_source": str(Path(args.autoregressive).resolve()),
        "direct_source": str(Path(args.direct).resolve()),
    }
    (output / "track2_hybrid_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
