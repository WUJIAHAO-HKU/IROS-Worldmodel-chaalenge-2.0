#!/usr/bin/env python3
"""Bind every effective seed file consumed by the local final-128 runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_seeds(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    seeds = payload["adjust_bottle"]["success_seeds"]
    return [int(seed) for seed in seeds]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen seed bundle: {args.output}")

    shard_manifest = args.eval_root / "seed_shards" / "manifest.json"
    regroup_manifest = args.eval_root / "seed_batches16" / "manifest.json"
    source = json.loads(shard_manifest.read_text(encoding="utf-8"))
    expected_selected = [int(seed) for seed in source["selected_seeds"]]
    specifications = [("00", 8, args.eval_root / "seed_shards" / "shard_00.json")]
    specifications.extend(
        (f"{index:02d}", 8 if index == 8 else 16,
         args.eval_root / "seed_batches16" / f"batch_{index:02d}.json")
        for index in range(1, 9)
    )

    batches = []
    effective_seeds: list[int] = []
    for batch, expected_count, path in specifications:
        if not path.is_file():
            raise FileNotFoundError(path)
        seeds = load_seeds(path)
        if len(seeds) != expected_count or len(set(seeds)) != expected_count:
            raise RuntimeError(f"invalid count or duplicates in effective batch {batch}")
        effective_seeds.extend(seeds)
        batches.append({
            "batch": batch,
            "count": expected_count,
            "path": str(path),
            "sha256": sha256(path),
            "seeds": seeds,
        })

    if effective_seeds != expected_selected:
        raise RuntimeError("effective batch sequence differs from official selected sequence")
    if len(effective_seeds) != 128 or len(set(effective_seeds)) != 128:
        raise RuntimeError("effective final seed bundle is not exactly 128 unique seeds")

    sequence_bytes = json.dumps(
        effective_seeds, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    payload = {
        "format": "strict-track2-effective-final128-seed-bundle-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "bind the exact nine seed files consumed by run_strict_track2_final128_eval.sh",
        "result_data_used": False,
        "seed_count": 128,
        "unique_seed_count": 128,
        "effective_seed_sequence_sha256": hashlib.sha256(sequence_bytes).hexdigest(),
        "official_partition_manifest": {
            "path": str(shard_manifest),
            "sha256": sha256(shard_manifest),
        },
        "hardware_regroup_manifest": {
            "path": str(regroup_manifest),
            "sha256": sha256(regroup_manifest),
        },
        "batches": batches,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
