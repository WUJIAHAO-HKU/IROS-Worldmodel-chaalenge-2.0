#!/usr/bin/env python3
"""Blend frozen parametric and public-retrieval predictions without labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--retrieval-cache", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 < args.alpha < 1.0:
        raise SystemExit("alpha must be inside (0,1)")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    with np.load(args.baseline_cache, allow_pickle=False) as values:
        baseline = values["baseline"].copy()
        paths = values["path"].astype(str)
    with np.load(args.retrieval_cache, allow_pickle=False) as values:
        retrieval = values["candidate"].copy()
        retrieval_paths = values["path"].astype(str)
    if not np.array_equal(paths, retrieval_paths) or baseline.shape != retrieval.shape:
        raise ValueError("baseline and retrieval caches do not align")
    candidate = np.clip(
        np.rint((1.0 - args.alpha) * baseline.astype(np.float32) + args.alpha * retrieval.astype(np.float32)),
        0,
        255,
    ).astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, candidate=candidate, path=paths)
    manifest = {
        "format": "strict-track2-v216-parametric-retrieval-blend-cache-v1",
        "baseline_cache": str(args.baseline_cache.resolve()),
        "baseline_cache_sha256": sha256(args.baseline_cache),
        "retrieval_cache": str(args.retrieval_cache.resolve()),
        "retrieval_cache_sha256": sha256(args.retrieval_cache),
        "alpha": args.alpha,
        "label_or_reward_used_for_blending": False,
        "hidden_or_final_data": False,
        "real_submission": False,
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
