#!/usr/bin/env python3
"""Verify frozen post-training tool hashes before an evaluation stage starts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--tool", action="append", required=True)
    args = parser.parse_args()
    record = json.loads(args.preregistration.read_text())
    if record.get("format") != (
        "strict-track2-v304-posttraining-pipeline-preregistration-v2"
    ):
        raise RuntimeError("unexpected post-training preregistration format")
    guards = record.get("guards", {})
    if not (
        guards.get("hidden_or_final_outcomes_used_for_training_or_selection")
        is False
        and guards.get("public_policy_outcomes_read_at_registration") is False
        and guards.get("real_submission") is False
        and guards.get("participant_component") == "world-model RGB service only"
        and guards.get("policy_mirroring") is False
        and guards.get("sft_co_training") is False
    ):
        raise RuntimeError("post-training preregistration guards failed")
    if record["final128_gate"] != {
        "count": 128,
        "one_unique_frozen_candidate": True,
        "selection_after_final": False,
        "successes_min": 85,
    }:
        raise RuntimeError("final128 target or one-candidate rule changed")
    for name in args.tool:
        if name not in record["tools"]:
            raise KeyError(f"unregistered tool: {name}")
        expected = record["tools"][name]
        path = Path(expected["path"])
        if not path.is_file() or sha256(path) != expected["sha256"]:
            raise RuntimeError(f"tool hash mismatch: {name}")
    print("V304_POSTTRAINING_TOOL_HASHES_VERIFIED " + " ".join(args.tool))


if __name__ == "__main__":
    main()
