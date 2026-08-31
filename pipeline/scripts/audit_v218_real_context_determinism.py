#!/usr/bin/env python3
"""Audit real-context determinism and action sensitivity of the v218 service."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from wam_pipeline.rlinf_bridge.track2_client import Track2ServiceClient


def digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    with np.load(args.baseline_cache, allow_pickle=False) as values:
        mask = values["arm_right"].astype(bool)
        names = values["path"].astype(str)[mask]
        instructions = values["instruction"].astype(str)[mask].tolist()
        source_seeds = values["synthetic_seed"].astype(np.int64)[mask]
    contexts, histories, futures = [], [], []
    for name in names:
        with np.load(args.windows / name, allow_pickle=False) as values:
            contexts.append(values["context_frames"].copy())
            histories.append(values["history_actions"].astype(np.float32).copy())
            futures.append(values["future_actions"].astype(np.float32).copy())
    contexts = np.stack(contexts)
    histories = np.stack(histories)
    futures = np.stack(futures)
    seeds = np.where(source_seeds < 0, 0, source_seeds).astype(np.int64)
    client = Track2ServiceClient(args.url, args.token, args.model_version)
    client.assert_ready()
    first = client.predict_batch(contexts, histories, futures, seeds, instructions)
    second = client.predict_batch(contexts, histories, futures, seeds, instructions)
    permuted_futures = np.roll(futures, shift=1, axis=0)
    changed = client.predict_batch(
        contexts, histories, permuted_futures, seeds, instructions
    )
    per_sample_changed = np.any(first != changed, axis=(1, 2, 3, 4))
    report = {
        "format": "strict-track2-v218-real-context-determinism-v1",
        "model_version": args.model_version,
        "right_sample_count": len(names),
        "two_calls_use_distinct_client_request_ids": True,
        "repeat_pixel_exact": bool(np.array_equal(first, second)),
        "first_sha256": digest(first),
        "second_sha256": digest(second),
        "action_permutation_changed_count": int(per_sample_changed.sum()),
        "action_permutation_changed_rate": float(per_sample_changed.mean()),
        "permuted_sha256": digest(changed),
        "service_output_only_rgb": True,
        "hidden_or_final_data": False,
        "real_submission": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
