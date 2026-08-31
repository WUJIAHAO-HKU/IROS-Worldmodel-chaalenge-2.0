#!/usr/bin/env python3
"""Create a self-contained, hash-verified Track 2 Wan/direct-flow ensemble."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


FORMAT = "track2-wan-direct-flow-ensemble-v1"
COMPONENTS = {
    "wan": ("track2_wan_lora.pt", "training_manifest.json", "action_normalization.npz"),
    "direct_flow": ("model.pt", "training_manifest.json", "action_normalization.npz", "track2_direct_flow_unet_config.npz"),
}
BASE_MANIFEST = "track2_wan_weight_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_component(source: Path, destination: Path, required: tuple[str, ...]) -> dict[str, str]:
    if not source.is_dir():
        raise ValueError(f"component directory is missing: {source}")
    destination.mkdir(parents=True, exist_ok=False)
    hashes: dict[str, str] = {}
    for filename in required:
        original = source / filename
        if not original.is_file():
            raise ValueError(f"component artifact is missing: {original}")
        copied = destination / filename
        shutil.copy2(original, copied)
        original_hash = sha256(original)
        if sha256(copied) != original_hash:
            raise RuntimeError(f"copy integrity check failed: {original}")
        hashes[filename] = original_hash
    return hashes


def base_identity(base_model: Path) -> dict[str, str]:
    manifest = base_model / BASE_MANIFEST
    if not manifest.is_file():
        raise ValueError(f"Wan base is missing verified weight manifest: {manifest}")
    try:
        contents = json.loads(manifest.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Wan base manifest is invalid JSON: {manifest}") from exc
    repository = contents.get("repository")
    revision = contents.get("revision")
    if not isinstance(repository, str) or not repository or not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("Wan base manifest has no immutable repository revision")
    return {
        "repository": repository,
        "revision": revision,
        "weight_manifest_sha256": sha256(manifest),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Package a fixed Wan/direct-flow RGB ensemble.")
    parser.add_argument("--wan-checkpoint", required=True)
    parser.add_argument("--direct-flow-checkpoint", required=True)
    parser.add_argument("--wan-base-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wan-weight", type=float, default=0.25)
    parser.add_argument("--direct-flow-weight", type=float, default=0.75)
    parser.add_argument("--wan-inference-steps", type=int, default=45)
    parser.add_argument("--wan-inference-solver", choices=("euler", "heun"), default="euler")
    args = parser.parse_args()
    if not 0.0 <= args.wan_weight <= 1.0 or not 0.0 <= args.direct_flow_weight <= 1.0:
        raise SystemExit("ensemble weights must be in [0, 1]")
    if abs(args.wan_weight + args.direct_flow_weight - 1.0) > 1e-8:
        raise SystemExit("ensemble weights must sum to one")
    if args.wan_inference_steps < 1:
        raise SystemExit("--wan-inference-steps must be positive")

    wan_source = Path(args.wan_checkpoint).resolve()
    flow_source = Path(args.direct_flow_checkpoint).resolve()
    wan_base = Path(args.wan_base_model).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing ensemble directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        wan_hashes = copy_component(wan_source, stage / "wan", COMPONENTS["wan"])
        flow_hashes = copy_component(flow_source, stage / "direct_flow", COMPONENTS["direct_flow"])
        config = {
            "format": FORMAT,
            "wan_weight": args.wan_weight,
            "direct_flow_weight": args.direct_flow_weight,
            "wan": {
                "directory": "wan",
                "inference_steps": args.wan_inference_steps,
                "inference_solver": args.wan_inference_solver,
                "base_model": base_identity(wan_base),
                "source_checkpoint": str(wan_source),
                "sha256": wan_hashes,
            },
            "direct_flow": {
                "directory": "direct_flow",
                "source_checkpoint": str(flow_source),
                "sha256": flow_hashes,
            },
        }
        (stage / "ensemble_config.json").write_text(json.dumps(config, indent=2) + "\n")
        os.replace(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(json.dumps({"status": "packaged", "checkpoint_dir": str(output), **config}, indent=2))


if __name__ == "__main__":
    main()
