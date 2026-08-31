#!/usr/bin/env python3
"""Verify parallel V15 service output against a frozen serial batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from wam_pipeline.rlinf_bridge.track2_client import Track2ServiceClient


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--golden-sha256", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--token", default="local-dev-token")
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--max-seconds", type=float, default=180.0)
    parser.add_argument("--warmup-first", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    actual_sha256 = sha256(args.golden)
    if actual_sha256 != args.golden_sha256:
        raise SystemExit("golden NPZ SHA256 mismatch")
    with np.load(args.golden, allow_pickle=False) as golden:
        context = golden["context_frames"]
        history = golden["history_actions"]
        future = golden["future_actions"]
        seeds = golden["seeds"]
        expected = golden["predicted_frames"]
        instructions = json.loads(str(golden["instructions_json"]))

    client = Track2ServiceClient(
        args.url, args.token, args.model_version, timeout_seconds=args.timeout
    )
    client.assert_ready()
    warmup_seconds = None
    warmup_exact_equal = None
    if args.warmup_first:
        warmup_started = time.perf_counter()
        warmup = client.predict_batch(
            context[:1], history[:1], future[:1], seeds[:1], instructions[:1]
        )
        warmup_seconds = time.perf_counter() - warmup_started
        warmup_exact_equal = bool(np.array_equal(warmup, expected[:1]))
    started = time.perf_counter()
    try:
        actual = client.predict_batch(
            context, history, future, seeds, instructions
        )
    except Exception as exc:
        response = getattr(exc, "response", None)
        if response is not None:
            print(response.text)
        raise
    elapsed = time.perf_counter() - started
    mismatch = np.flatnonzero(actual.reshape(-1) != expected.reshape(-1))
    report = {
        "format": "strict-track2-v157-parallel-batch-benchmark-v1",
        "golden_npz": str(args.golden),
        "golden_npz_sha256": actual_sha256,
        "batch_size": int(actual.shape[0]),
        "compared_rgb_bytes": int(actual.size),
        "warmup_seconds": warmup_seconds,
        "warmup_exact_equal": warmup_exact_equal,
        "exact_equal": bool(mismatch.size == 0),
        "mismatched_rgb_bytes": int(mismatch.size),
        "first_mismatch_flat_index": None if mismatch.size == 0 else int(mismatch[0]),
        "elapsed_seconds": elapsed,
        "maximum_seconds": args.max_seconds,
        "latency_passed": elapsed <= args.max_seconds,
        "passed": bool(
            mismatch.size == 0
            and elapsed <= args.max_seconds
            and warmup_exact_equal is not False
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
