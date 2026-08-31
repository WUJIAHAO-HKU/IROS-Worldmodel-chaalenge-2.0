#!/usr/bin/env python3
"""Validate and aggregate exact task instructions recovered by seed probes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", required=True)
    parser.add_argument("--seed-manifest", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    expected_document = json.loads(Path(args.seed_manifest).read_text())
    expected = sorted(int(seed) for seed in expected_document["selected_seeds"])
    rows = []
    source_hashes = {}
    for path in sorted(Path(args.probe_root).glob("batch_*/eval/complete.json")):
        raw = path.read_bytes()
        source_hashes[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
        document = json.loads(raw)
        if document.get("format") != "strict-track2-instruction-probe-v1":
            raise ValueError(f"unsupported probe: {path}")
        rows.extend(document["rows"])
    mapping = {int(row["seed"]): str(row["task_instruction"]) for row in rows}
    arms = {int(row["seed"]): bool(row["arm_right"]) for row in rows}
    if len(mapping) != len(rows):
        raise ValueError("duplicate seeds in instruction probes")
    if sorted(mapping) != expected:
        raise ValueError(
            f"probe seed mismatch: missing={sorted(set(expected)-set(mapping))}, "
            f"extra={sorted(set(mapping)-set(expected))}"
        )
    if any(not value.strip() for value in mapping.values()):
        raise ValueError("empty recovered instruction")

    window_arms = {}
    for path in Path(args.windows).glob("episode*_*.npz"):
        with np.load(path, allow_pickle=False) as values:
            seed = int(values["synthetic_seed"])
            arm = bool(values["arm_right"])
        if seed in window_arms and window_arms[seed] != arm:
            raise ValueError(f"window arm changes for seed {seed}")
        window_arms[seed] = arm
    if sorted(window_arms) != expected:
        raise ValueError("window seeds do not match seed manifest")
    mismatched_arms = [seed for seed in expected if window_arms[seed] != arms[seed]]
    if mismatched_arms:
        raise ValueError(f"probe/window arm mismatch: {mismatched_arms}")

    canonical = json.dumps({str(seed): mapping[seed] for seed in expected}, sort_keys=True).encode()
    report = {
        "format": "strict-track2-exact-instruction-map-v1",
        "purpose": "frozen official reward-model conditioning for offline parent training and validation",
        "seed_count": len(expected),
        "unique_instruction_count": len(set(mapping.values())),
        "left_count": sum(not arms[seed] for seed in expected),
        "right_count": sum(arms[seed] for seed in expected),
        "seed_to_instruction": {str(seed): mapping[seed] for seed in expected},
        "seed_to_arm_right": {str(seed): arms[seed] for seed in expected},
        "mapping_sha256": hashlib.sha256(canonical).hexdigest(),
        "probe_source_sha256": source_hashes,
        "seed_manifest": str(Path(args.seed_manifest).resolve()),
        "windows": str(Path(args.windows).resolve()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("seed_count", "unique_instruction_count", "left_count", "right_count", "mapping_sha256")}))


if __name__ == "__main__":
    main()
