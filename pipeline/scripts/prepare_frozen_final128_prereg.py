#!/usr/bin/env python3
"""Authorize exactly one local final-128 evaluation after unique freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-manifest", required=True, type=Path)
    parser.add_argument("--final-seed-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    freeze = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    if freeze.get("unique_final_candidate") is not True:
        raise RuntimeError("candidate was not uniquely frozen")
    checkpoint = Path(freeze["checkpoint"])
    if sha256(checkpoint) != freeze["checkpoint_sha256"]:
        raise RuntimeError("frozen checkpoint hash mismatch")
    if args.output.exists() or args.output_root.exists():
        raise RuntimeError("final-128 preregistration or output root already exists")

    payload = {
        "format": "strict-track2-frozen-local-final128-preregistration-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "authorized": True,
        "scope": "one local RoboTwin final-128 evaluation of the uniquely frozen candidate",
        "contest_submission": False,
        "variant": freeze["variant"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": freeze["checkpoint_sha256"],
        "freeze_manifest": str(args.freeze_manifest),
        "freeze_manifest_sha256": sha256(args.freeze_manifest),
        "final_seed_manifest": str(args.final_seed_manifest),
        "final_seed_manifest_sha256": sha256(args.final_seed_manifest),
        "output_root": str(args.output_root),
        "target": {"successes_min": 85, "count": 128},
        "selection_after_this_evaluation": False,
        "retry_policy": "resume incomplete batches only; never evaluate another candidate",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
