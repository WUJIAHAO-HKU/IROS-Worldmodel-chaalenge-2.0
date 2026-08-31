#!/usr/bin/env python3
"""Package the accepted bounded post-V15 residual as an immutable release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


FORMAT = "track2-v15-post-residual-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-release", required=True)
    parser.add_argument("--residual-checkpoint", required=True)
    parser.add_argument("--source-gate", required=True)
    parser.add_argument("--advantage-mask", required=True)
    parser.add_argument("--screen-report", required=True)
    parser.add_argument("--preregistration", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-strength", type=float, default=1.5)
    parser.add_argument("--maximum-deployed-residual-255", type=float, default=8.0)
    parser.add_argument("--model-version", default="track2-v15.4-post-residual")
    args = parser.parse_args()
    base = Path(args.base_release).resolve()
    checkpoint = Path(args.residual_checkpoint).resolve()
    source_gate = Path(args.source_gate).resolve()
    mask = Path(args.advantage_mask).resolve()
    screen = Path(args.screen_report).resolve()
    preregistrations = [Path(value).resolve() for value in args.preregistration]
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite packaged release: {output}")
    if json.loads(screen.read_text()).get("passed") is not True:
        raise RuntimeError("post-V15 residual did not pass the preregistered screen")
    shutil.copytree(base, output, copy_function=os.link)
    adaptation = output / "post_v15_residual"
    if adaptation.exists():
        shutil.rmtree(adaptation)
    artifact_sources = {
        "post_v15_residual.pt": checkpoint / "post_v15_residual.pt",
        "residual_training_manifest.json": checkpoint / "training_manifest.json",
        "source_gate.pt": source_gate,
        "advantage_mask.npz": mask,
        "parent_screen_report.json": screen,
    }
    for index, path in enumerate(preregistrations):
        artifact_sources[f"preregistration_{index:02d}.json"] = path
    for relative, source in artifact_sources.items():
        link(source, adaptation / relative)
    manifest = {
        "format": FORMAT,
        "model_version": args.model_version,
        "base_release": str(base),
        "base_release_manifest_sha256": sha256(output / "release_manifest.json"),
        "output_strength": args.output_strength,
        "maximum_deployed_residual_255": args.maximum_deployed_residual_255,
        "routing": "last-context visual source gate plus raw-action-delta active arm",
        "router_uses_only_request_inputs": True,
        "official_left_protection": "zero residual mask",
        "inference_uses_target_or_reward": False,
        "screen_report": str(screen),
        "sha256": {
            relative: sha256(adaptation / relative) for relative in artifact_sources
        },
    }
    (adaptation / "adaptation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
