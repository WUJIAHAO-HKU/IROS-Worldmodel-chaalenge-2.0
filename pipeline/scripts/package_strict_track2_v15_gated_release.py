#!/usr/bin/env python3
"""Package the selected on-policy AR expert and visual gate over frozen V15."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


FORMAT = "track2-v15-gated-onpolicy-adaptation-v1"
AR_FILES = (
    "model.pt",
    "action_normalization.npz",
    "track2_autoregressive_unet_config.npz",
    "training_manifest.json",
)


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
    parser.add_argument("--adapted-autoregressive", required=True)
    parser.add_argument("--source-gate", required=True)
    parser.add_argument("--screen-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-version", default="track2-v15.1-gated-onpolicy-pilot")
    args = parser.parse_args()
    base, adapted, source_gate, screen = map(
        lambda value: Path(value).resolve(),
        (args.base_release, args.adapted_autoregressive, args.source_gate, args.screen_report),
    )
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite packaged release: {output}")
    if json.loads(screen.read_text()).get("passed") is not True:
        raise RuntimeError("parent candidate did not pass the preregistered screen")
    shutil.copytree(base, output, copy_function=os.link)
    root = output / "onpolicy_adaptation"
    for filename in AR_FILES:
        link(adapted / filename, root / "adapted_autoregressive" / filename)
    link(source_gate / "source_gate.pt", root / "source_gate.pt")
    link(source_gate / "training_manifest.json", root / "source_gate_training_manifest.json")
    link(screen, root / "parent_screen_report.json")
    relative_files = [
        *(f"adapted_autoregressive/{filename}" for filename in AR_FILES),
        "source_gate.pt",
        "source_gate_training_manifest.json",
        "parent_screen_report.json",
    ]
    manifest = {
        "format": FORMAT,
        "model_version": args.model_version,
        "base_release": str(base),
        "base_release_manifest_sha256": sha256(output / "release_manifest.json"),
        "adapted_autoregressive": str(adapted),
        "source_gate": str(source_gate),
        "screen_report": str(screen),
        "screen_report_sha256": sha256(screen),
        "hard_route_threshold": .5,
        "router_uses_only_real_context_rgb": True,
        "sha256": {relative: sha256(root / relative) for relative in relative_files},
    }
    (root / "adaptation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
