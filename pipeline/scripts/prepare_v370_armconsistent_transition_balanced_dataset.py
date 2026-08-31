#!/usr/bin/env python3
"""Pre-register the immutable public-only v370 dataset construction."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--auditor", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source_summary = json.loads(args.source_summary.read_text())
    split = json.loads(args.split.read_text())
    if source_summary["episodes"] != split["train_episodes"] or len(split["train_episodes"]) != 40:
        raise ValueError("source conversion and frozen public split disagree")

    registration = {
        "format": "strict-track2-v370-dataset-preregistration-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "fix task/arm label conflicts and emphasize real right post-grasp motion",
        "immutable_inputs": {
            "source_summary": str(args.source_summary.resolve()),
            "source_summary_sha256": sha256(args.source_summary),
            "split": str(args.split.resolve()),
            "split_sha256": sha256(args.split),
            "builder": str(args.builder.resolve()),
            "builder_sha256": sha256(args.builder),
            "auditor": str(args.auditor.resolve()),
            "auditor_sha256": sha256(args.auditor),
        },
        "dataset": str(args.dataset.resolve()),
        "fixed_recipe": {
            "left": "25 public train trajectories, full t=1..end-1, one copy",
            "right": "15 public train trajectories, first_close-4 through first_close+48, four copies",
            "task": "v205 action-arm-consistent episode_to_instruction; explicit arm required",
            "state_alignment": "state[t]=joint_action[t-1], action[t]=joint_action[t]",
            "image_writer_threads": 4,
        },
        "acceptance": [
            "only the frozen public train40 source episodes are read",
            "25 unique left and 15 unique right sources",
            "85 dataset episodes and each right source repeated exactly four times",
            "every task explicitly names exactly the action-derived arm",
            "all state/action rows reconstruct bit-exactly from source",
            "image and parquet row counts match",
        ],
        "reserved_or_hidden_evaluation_access": False,
        "real_competition_submission": False,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "preregistration.json").write_text(json.dumps(registration, indent=2) + "\n")
    print(json.dumps(registration, indent=2))


if __name__ == "__main__":
    main()
