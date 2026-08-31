#!/usr/bin/env python3
"""Preflight immutable local final128 seeds and runner without reading outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_seeds(path: Path) -> list[int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [int(seed) for seed in data["adjust_bottle"]["success_seeds"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    rows: list[int] = []
    batch_checks = []
    for item in bundle["batches"]:
        path = Path(item["path"])
        seeds = load_seeds(path)
        checks = {
            "count": len(seeds) == int(item["count"]),
            "unique_within_batch": len(seeds) == len(set(seeds)),
            "sha256": sha256(path) == item["sha256"],
        }
        batch_checks.append({"batch": item["batch"], "path": str(path), "checks": checks})
        rows.extend(seeds)

    partition = bundle["official_partition_manifest"]
    partition_path = Path(partition["path"])
    source = json.loads(partition_path.read_text(encoding="utf-8"))
    expected = [int(seed) for seed in source["selected_seeds"]]
    runner_text = args.runner.read_text(encoding="utf-8")
    forbidden = [
        line.strip() for line in runner_text.splitlines()
        if re.search(r"\b(curl|wget|requests\.|urllib|submit|submission)\b", line, re.I)
    ]
    checks = {
        "bundle_result_data_unused": bundle.get("result_data_used") is False,
        "bundle_seed_count_128": bundle.get("seed_count") == 128,
        "effective_count_128": len(rows) == 128,
        "effective_unique_128": len(set(rows)) == 128,
        "effective_sequence_matches_partition": rows == expected,
        "partition_manifest_hash": sha256(partition_path) == partition["sha256"],
        "runner_exact_batches": "TRACK2_INSTRUMENTED_BATCHES:-00 01 02 03 04 05 06 07 08" in runner_text,
        "runner_candidate_checkpoint_required": "TRACK2_FINAL128_CHECKPOINT:?" in runner_text,
        "runner_candidate_variant_required": "TRACK2_FINAL128_VARIANT:?" in runner_text,
        "runner_can_disable_baseline": "TRACK2_FINAL128_RUN_BASELINE" in runner_text,
        "runner_no_network_or_submission_tokens": not forbidden,
        "all_batch_files_valid": all(all(row["checks"].values()) for row in batch_checks),
    }
    report = {
        "format": "strict-track2-v277-final128-static-preflight-v1",
        "bundle": str(args.bundle),
        "bundle_sha256": sha256(args.bundle),
        "runner": str(args.runner),
        "runner_sha256": sha256(args.runner),
        "effective_seed_count": len(rows),
        "effective_unique_seed_count": len(set(rows)),
        "batch_counts": [len(load_seeds(Path(item["path"]))) for item in bundle["batches"]],
        "forbidden_runner_matches": forbidden,
        "batch_checks": batch_checks,
        "checks": checks,
        "passed": all(checks.values()),
        "outcomes_read": False,
        "official_submission": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
