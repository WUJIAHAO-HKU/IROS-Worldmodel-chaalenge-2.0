#!/usr/bin/env python3
"""Export 128-frame recursive predictions through the real Track-2 HTTP API."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from wam_pipeline.rlinf_bridge.track2_client import Track2ServiceClient


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_name(name: str) -> tuple[int, int]:
    stem = Path(name).stem
    episode_text, start_text = stem.split("_")
    return int(episode_text[7:]), int(start_text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--query-windows", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunks", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.chunks < 1 or not 1 <= args.batch_size <= 8:
        raise SystemExit("chunks must be positive and batch-size must be in [1,8]")

    with np.load(args.baseline_cache, allow_pickle=False) as values:
        paths = values["path"].astype(str)
        source_seeds = values["synthetic_seed"].astype(np.int64)
        # Public demonstrations use -1 as a local "missing" sentinel, while
        # the official HTTP contract accepts only non-negative int64 seeds.
        # This backend's native batch path is deterministic and discards seed.
        seeds = np.where(source_seeds < 0, 0, source_seeds).astype(np.int64)
        instructions = values["instruction"].astype(str)
    query_index = {
        parse_name(path.name): path
        for path in args.query_windows.glob("episode*_*.npz")
    }
    contexts, histories, all_future = [], [], []
    for name in paths:
        episode, start = parse_name(name)
        first = query_index[(episode, start)]
        with np.load(first, allow_pickle=False) as values:
            contexts.append(values["context_frames"].copy())
            histories.append(values["history_actions"].astype(np.float32).copy())
        chunks = []
        for chunk in range(args.chunks):
            path = query_index[(episode, start + 8 * chunk)]
            with np.load(path, allow_pickle=False) as values:
                chunks.append(values["future_actions"].astype(np.float32).copy())
        all_future.append(chunks)
    contexts = np.stack(contexts)
    histories = np.stack(histories)
    all_future = np.asarray(all_future)

    client = Track2ServiceClient(args.url, args.token, args.model_version)
    client.assert_ready()
    predicted_chunks = [[] for _ in paths]
    for chunk in range(args.chunks):
        actions = all_future[:, chunk]
        for begin in range(0, len(paths), args.batch_size):
            end = min(begin + args.batch_size, len(paths))
            predicted = client.predict_batch(
                contexts[begin:end],
                histories[begin:end],
                actions[begin:end],
                seeds[begin:end] + chunk,
                instructions[begin:end].tolist(),
            )
            for offset, frames in enumerate(predicted):
                predicted_chunks[begin + offset].append(frames)
            contexts[begin:end] = np.concatenate(
                (contexts[begin:end], predicted), axis=1
            )[:, -5:]
            histories[begin:end] = np.concatenate(
                (histories[begin:end], actions[begin:end]), axis=1
            )[:, -4:]
        print(
            json.dumps({"completed_chunk": chunk + 1, "chunks": args.chunks}),
            flush=True,
        )
    candidate = np.stack([np.concatenate(row, axis=0) for row in predicted_chunks])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, candidate=candidate, path=paths)
    manifest = {
        "format": "strict-track2-v217-online-recursive-service-cache-v1",
        "baseline_cache": str(args.baseline_cache.resolve()),
        "baseline_cache_sha256": sha256(args.baseline_cache),
        "query_windows": str(args.query_windows.resolve()),
        "service_url": args.url,
        "model_version": args.model_version,
        "sequence_count": len(paths),
        "chunks": args.chunks,
        "batch_size": args.batch_size,
        "missing_source_seed_sentinel": -1,
        "missing_source_seed_replacement": 0,
        "replaced_seed_count": int((source_seeds < 0).sum()),
        "predicted_context_between_chunks": True,
        "request_outputs_only_rgb": True,
        "hidden_or_final_data": False,
        "real_submission": False,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
