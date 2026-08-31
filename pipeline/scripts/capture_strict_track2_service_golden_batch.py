#!/usr/bin/env python3
"""Capture a real-context service batch for byte-exact runtime optimization gates."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import requests

from wam_pipeline.rlinf_bridge.track2_client import Track2ServiceClient


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--token", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--retry-seconds", type=float, default=0.5)
    parser.add_argument("--max-wait-seconds", type=float, default=900.0)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    if args.output_npz.exists() or args.output_json.exists():
        raise SystemExit("refusing to overwrite an existing capture")
    with np.load(args.golden, allow_pickle=False) as source:
        stop = args.begin + args.count
        context = source["context_frames"][args.begin:stop]
        history = source["history_actions"][args.begin:stop]
        future = source["future_actions"][args.begin:stop]
        seeds = source["seeds"][args.begin:stop]
        instructions = json.loads(str(source["instructions_json"]))[args.begin:stop]
    if len(context) != args.count:
        raise SystemExit("requested slice exceeds the golden batch")

    client = Track2ServiceClient(args.url, args.token, args.model_version)
    client.assert_ready()
    deadline = time.monotonic() + args.max_wait_seconds
    attempts = 0
    started = time.perf_counter()
    while True:
        attempts += 1
        try:
            predicted = client.predict_batch(
                context, history, future, seeds, instructions
            )
            break
        except requests.HTTPError as exc:
            if exc.response is None or exc.response.status_code != 429:
                raise
            if time.monotonic() >= deadline:
                raise TimeoutError("service remained overloaded") from exc
            time.sleep(args.retry_seconds)
    elapsed = time.perf_counter() - started

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_npz.with_suffix(args.output_npz.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, predicted_frames=predicted)
    temporary.replace(args.output_npz)
    report = {
        "format": "strict-track2-real-context-service-golden-batch-v1",
        "captured_at": datetime.datetime.now().astimezone().isoformat(),
        "source_npz": str(args.golden),
        "source_npz_sha256": file_sha256(args.golden),
        "slice_begin": args.begin,
        "slice_count": args.count,
        "model_version": args.model_version,
        "attempts_including_429": attempts,
        "wall_seconds_including_overload_retries": elapsed,
        "prediction_shape": list(predicted.shape),
        "prediction_dtype": str(predicted.dtype),
        "prediction_sha256": array_sha256(predicted),
        "capture_npz": str(args.output_npz),
        "capture_npz_sha256": file_sha256(args.output_npz),
    }
    json_temporary = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    json_temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    json_temporary.replace(args.output_json)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
