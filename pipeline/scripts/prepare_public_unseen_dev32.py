#!/usr/bin/env python3
"""Pre-register an unseen, public-only RoboTwin policy development split.

The split is chosen deterministically from the public training seed list.  It
excludes every seed used by the 128-episode on-policy world-model corpus and
every official evaluation seed.  The script is intentionally outcome-blind.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def task_seeds(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    task = payload["adjust_bottle"]
    return [int(seed) for seed in task["success_seeds"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-seeds", required=True, type=Path)
    parser.add_argument("--used-manifest", required=True, type=Path)
    parser.add_argument("--eval-seeds", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if args.count <= 0 or args.batch_size <= 0:
        raise ValueError("count and batch-size must be positive")

    train = task_seeds(args.train_seeds)
    used_payload = json.loads(args.used_manifest.read_text(encoding="utf-8"))
    used = {int(seed) for seed in used_payload["selected_seeds"]}
    official_eval = set(task_seeds(args.eval_seeds))
    selected = [seed for seed in train if seed not in used and seed not in official_eval][
        : args.count
    ]
    if len(selected) != args.count:
        raise RuntimeError(f"only {len(selected)} eligible seeds for requested {args.count}")
    if len(set(selected)) != len(selected):
        raise RuntimeError("selected seeds are not unique")

    args.output_root.mkdir(parents=True, exist_ok=False)
    batches: list[dict[str, object]] = []
    for batch_id, offset in enumerate(range(0, len(selected), args.batch_size)):
        batch_seeds = selected[offset : offset + args.batch_size]
        batch_path = args.output_root / f"batch_{batch_id:02d}.json"
        write_json(
            batch_path,
            {
                "adjust_bottle": {
                    "success_seeds": batch_seeds,
                    "task_name": "adjust_bottle",
                }
            },
        )
        batches.append(
            {
                "batch": batch_id,
                "count": len(batch_seeds),
                "path": batch_path.name,
                "seeds": batch_seeds,
                "sha256": sha256(batch_path),
            }
        )

    manifest = {
        "format": "strict-track2-public-unseen-policy-dev-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "public local RoboTwin policy development screening only",
        "selection_rule": (
            "first N adjust_bottle public train success seeds in source order, "
            "excluding the frozen 128-episode world-model corpus and all official eval seeds"
        ),
        "selection_uses_policy_outcomes": False,
        "source": {
            "public_train_seeds": str(args.train_seeds),
            "public_train_seeds_sha256": sha256(args.train_seeds),
            "used_world_model_manifest": str(args.used_manifest),
            "used_world_model_manifest_sha256": sha256(args.used_manifest),
            "official_eval_seeds_exclusion_only": str(args.eval_seeds),
            "official_eval_seeds_sha256": sha256(args.eval_seeds),
        },
        "count": len(selected),
        "selected_seeds": selected,
        "intersection_with_world_model_corpus": sorted(set(selected) & used),
        "intersection_with_official_eval_seeds": sorted(set(selected) & official_eval),
        "batches": batches,
        "prohibitions": [
            "not for world-model training",
            "not for reward calibration",
            "not an official submission",
            "not the reserved final 128 evaluation",
        ],
    }
    write_json(args.output_root / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
