#!/usr/bin/env python3
"""Validate and fingerprint an official RLinf Wan RobotWin checkpoint directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wam_pipeline.official_rlinf_wan import (
    OFFICIAL_RLINF_WAN_RUNTIME_REVISION,
    sha256_file,
    validate_official_rlinf_wan_checkpoint,
    validate_official_rlinf_wan_runtime,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument(
        "--diffsynth-root",
        help="Optional isolated DiffSynth checkout to pin alongside the checkpoint manifest.",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Exact organizer-published model page, revision, or archive URL used to obtain this checkpoint.",
    )
    parser.add_argument(
        "--output",
        help="Defaults to official_rlinf_wan_checkpoint_manifest.json in --checkpoint-dir.",
    )
    parser.add_argument(
        "--skip-sha256",
        action="store_true",
        help="Check the required layout only; do not write a provenance manifest.",
    )
    args = parser.parse_args()
    checkpoint = validate_official_rlinf_wan_checkpoint(args.checkpoint_dir)
    runtime = validate_official_rlinf_wan_runtime(args.diffsynth_root) if args.diffsynth_root else None
    if args.skip_sha256:
        print(f"official RLinf Wan layout is valid: {checkpoint.root}")
        return

    dataset_files = sorted(checkpoint.dataset.glob("*.npy"))
    manifest = {
        "format": "official-rlinf-wan-checkpoint-v1",
        "checkpoint_dir": str(checkpoint.root),
        "source": args.source,
        "diffsynth_runtime_revision": OFFICIAL_RLINF_WAN_RUNTIME_REVISION,
        "diffsynth_runtime": str(runtime) if runtime else None,
        "files": {
            "dit_model.safetensors": {
                "bytes": checkpoint.dit_model.stat().st_size,
                "sha256": sha256_file(checkpoint.dit_model),
            },
            "Wan2.2_VAE.pth": {
                "bytes": checkpoint.vae.stat().st_size,
                "sha256": sha256_file(checkpoint.vae),
            },
        },
        "dataset": {
            "directory": str(checkpoint.dataset),
            "trajectory_npy_count": len(dataset_files),
            "trajectory_npy_names": [path.name for path in dataset_files],
        },
    }
    output = Path(args.output) if args.output else checkpoint.root / "official_rlinf_wan_checkpoint_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
