#!/usr/bin/env python3
"""Recompute public112 split provenance before any candidate outcome is read."""

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


def task_seeds(path: Path) -> list[int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [int(seed) for seed in data["adjust_bottle"]["success_seeds"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    source = manifest["source"]
    train_path = Path(source["public_train_seeds"])
    eval_path = Path(source["official_eval_seeds_exclusion_only"])
    world_path = Path(source["used_world_model_manifest"])
    train = set(task_seeds(train_path))
    official_eval = set(task_seeds(eval_path))
    world = set(map(int, json.loads(world_path.read_text(encoding="utf-8"))["selected_seeds"]))
    selected = list(map(int, manifest["selected_seeds"]))
    selected_set = set(selected)

    batches: list[int] = []
    batch_checks = []
    for item in manifest["batches"]:
        path = args.manifest.parent / item["path"]
        seeds = task_seeds(path)
        checks = {
            "count_16": len(seeds) == 16 == int(item["count"]),
            "sha256": sha256(path) == item["sha256"],
            "manifest_seed_list": seeds == list(map(int, item["seeds"])),
            "unique_within_batch": len(seeds) == len(set(seeds)),
        }
        batch_checks.append({"batch": int(item["batch"]), "path": str(path), "checks": checks})
        batches.extend(seeds)

    runner_text = args.runner.read_text(encoding="utf-8")
    forbidden = [
        line.strip() for line in runner_text.splitlines()
        if re.search(r"\b(curl|wget|requests\.|urllib|submit|submission)\b", line, re.I)
    ]
    checks = {
        "manifest_count_112": int(manifest["count"]) == 112,
        "selected_count_112": len(selected) == 112,
        "selected_unique_112": len(selected_set) == 112,
        "seven_batches": len(manifest["batches"]) == 7,
        "batch_sequence_matches_selected": batches == selected,
        "all_batch_files_valid": all(all(row["checks"].values()) for row in batch_checks),
        "selected_is_public_train_subset": selected_set <= train,
        "disjoint_from_official_eval": not (selected_set & official_eval),
        "disjoint_from_world_model_corpus": not (selected_set & world),
        "train_source_hash": sha256(train_path) == source["public_train_seeds_sha256"],
        "eval_exclusion_source_hash": sha256(eval_path) == source["official_eval_seeds_sha256"],
        "world_manifest_hash": sha256(world_path) == source["used_world_model_manifest_sha256"],
        "selection_did_not_use_policy_outcomes": manifest["selection_uses_policy_outcomes"] is False,
        "runner_has_single_batch_filter": "TRACK2_DEV_BATCH_FILTER" in runner_text,
        "runner_uses_local_seed_path": 'env.eval.seeds_path="$seed_path"' in runner_text,
        "runner_no_network_or_submission_tokens": not forbidden,
    }
    report = {
        "format": "strict-track2-v275-public112-static-preflight-v1",
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "runner": str(args.runner),
        "runner_sha256": sha256(args.runner),
        "selected_count": len(selected),
        "selected_unique_count": len(selected_set),
        "intersection_with_official_eval": sorted(selected_set & official_eval),
        "intersection_with_world_model_corpus": sorted(selected_set & world),
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
