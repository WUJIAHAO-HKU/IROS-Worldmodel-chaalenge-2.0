#!/usr/bin/env python3
"""Create deterministic candidate-only caches for a preregistered blend sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def tag(strength: float) -> str:
    return f"alpha_{strength:.4f}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()
    preregistration_path = Path(args.preregistration).resolve()
    preregistration = json.loads(preregistration_path.read_text())
    base_path = Path(preregistration["base_cache"]).resolve()
    candidate_path = Path(preregistration["candidate_cache"]).resolve()
    with np.load(base_path, allow_pickle=False) as base_values:
        baseline = base_values[preregistration.get("base_field", "baseline")].astype(np.float32)
        paths = base_values["path"].astype(str)
    with np.load(candidate_path, allow_pickle=False) as candidate_values:
        candidate = candidate_values[preregistration.get("candidate_field", "candidate")].astype(np.float32)
        candidate_paths = candidate_values["path"].astype(str)
    if paths.tolist() != candidate_paths.tolist() or baseline.shape != candidate.shape:
        raise ValueError("baseline and candidate caches are not paired")
    output_root = Path(args.output_directory)
    output_root.mkdir(parents=True, exist_ok=False)
    rows = []
    for strength in preregistration["screen_strengths"]:
        strength = float(strength)
        blended = np.clip(np.rint(baseline + strength * (candidate - baseline)), 0, 255).astype(np.uint8)
        output = output_root / f"{tag(strength)}.npz"
        np.savez_compressed(output, candidate=blended, path=paths)
        rows.append({"strength": strength, "cache": str(output.resolve())})
    manifest = {
        "format": "strict-track2-terminal-reward-blend-sweep-cache-v1",
        "preregistration": str(preregistration_path),
        "base_cache": str(base_path),
        "candidate_cache": str(candidate_path),
        "rows": rows,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
