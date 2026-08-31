#!/usr/bin/env python3
"""Create the preregistered bit-exact official-left protection mask."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.input)
    preregistration = Path(args.preregistration)
    with np.load(source, allow_pickle=False) as values:
        if str(values["format"]) != "strict-track2-residual-advantage-mask-v1":
            raise ValueError("unsupported advantage mask")
        mask = values["mask"].copy()
        threshold = values["beneficial_fraction_threshold"].copy()
        counts = values["stratum_count"].copy()
        config = values["checkpoint_config"].copy()
    mask[0, 0] = 0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        format=np.asarray("strict-track2-residual-advantage-mask-v1"),
        mask=mask,
        beneficial_fraction_threshold=threshold,
        official_left_protected=np.asarray(True),
        stratum_count=counts,
        coverage=mask.mean(axis=(2, 3, 4)).astype(np.float32),
        checkpoint_config=config,
    )
    manifest = {
        "format": "strict-track2-official-left-protected-advantage-mask-v1",
        "input": str(source.resolve()),
        "input_sha256": sha256(source),
        "preregistration": str(preregistration.resolve()),
        "preregistration_sha256": sha256(preregistration),
        "output": str(output.resolve()),
        "output_sha256": sha256(output),
        "coverage_by_source_arm": mask.mean(axis=(2, 3, 4)).tolist(),
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
