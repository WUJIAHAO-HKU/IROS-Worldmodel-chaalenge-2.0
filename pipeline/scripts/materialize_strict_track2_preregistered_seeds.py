#!/usr/bin/env python3
"""Materialize immutable RoboTwin seed batches from a Track 2 preregistration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def canonical(document: object) -> bytes:
    return (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_json(path: Path, document: object) -> str:
    payload = json.dumps(document, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)
    return hashlib.sha256(payload.encode()).hexdigest()


def batches(seeds: list[int], size: int) -> list[list[int]]:
    return [seeds[start : start + size] for start in range(0, len(seeds), size)]


def materialize(root: Path, name: str, seeds: list[int], batch_size: int) -> dict[str, object]:
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"duplicate seeds in {name}")
    entries = []
    for index, seed_batch in enumerate(batches(seeds, batch_size)):
        document = {
            "adjust_bottle": {
                "task_name": "adjust_bottle",
                "success_seeds": seed_batch,
            }
        }
        path = root / name / f"batch_{index:02d}.json"
        sha256 = write_json(path, document)
        entries.append({
            "batch": index,
            "path": str(path.resolve()),
            "count": len(seed_batch),
            "sha256": sha256,
        })
    manifest = {
        "format": "strict-track2-preregistered-robotwin-seeds-v1",
        "name": name,
        "count": len(seeds),
        "batch_size": batch_size,
        "seeds_sha256": hashlib.sha256(canonical(seeds)).hexdigest(),
        "batches": entries,
    }
    manifest_path = root / name / "manifest.json"
    manifest["manifest_path"] = str(manifest_path.resolve())
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")

    preregistration_path = Path(args.preregistration)
    preregistration = json.loads(preregistration_path.read_text())
    development = [int(seed) for seed in preregistration["new_policy_development32"]["seeds"]]
    acceptance = [int(seed) for seed in preregistration["new_final_acceptance128"]["seeds"]]
    overlap = sorted(set(development).intersection(acceptance))
    if overlap:
        raise ValueError(f"development and acceptance overlap: {overlap}")

    output_root = Path(args.output_root)
    result = {
        "format": "strict-track2-preregistered-seed-materialization-v1",
        "preregistration": str(preregistration_path.resolve()),
        "preregistration_sha256": hashlib.sha256(preregistration_path.read_bytes()).hexdigest(),
        "development": materialize(output_root, "development32", development, args.batch_size),
        "acceptance": materialize(output_root, "acceptance128", acceptance, args.batch_size),
        "development_acceptance_overlap": overlap,
    }
    output = output_root / "manifest.json"
    write_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
