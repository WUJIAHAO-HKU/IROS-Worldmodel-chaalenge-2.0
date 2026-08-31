#!/usr/bin/env python3
"""Export only candidate rollouts aligned to an immutable P2 baseline cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from export_strict_track2_p2_reward_cache import predict_sequence
from train_multichunk_reward_aligned_autoregressive_unet import CHUNK_FRAMES, MultiChunkWindows
from wam_pipeline.autoregressive_unet_runtime import Track2AutoregressiveUNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-cache", required=True, type=Path)
    parser.add_argument("--windows", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--episodes-key", default="validation_episodes")
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunks", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite candidate cache")

    with np.load(args.baseline_cache, allow_pickle=False) as values:
        paths = values["path"].astype(str).tolist()
    split = json.loads(args.split_manifest.read_text())
    dataset = MultiChunkWindows(
        args.windows,
        split[args.episodes_key],
        chunks=args.chunks,
        chunk_stride=CHUNK_FRAMES,
        arm_filter="all",
    )
    index_by_path = {sequence[0].name: index for index, sequence in enumerate(dataset.sequences)}
    missing = [path for path in paths if path not in index_by_path]
    if missing:
        raise ValueError(f"baseline paths missing from candidate dataset: {missing[:5]}")

    runtime = Track2AutoregressiveUNet(str(args.candidate), args.device)
    predictions = []
    for position, path in enumerate(paths, 1):
        item = dataset[index_by_path[path]]
        context_t, history_t, future_t = item[:3]
        predictions.append(
            predict_sequence(
                runtime,
                context_t.numpy(),
                history_t.numpy(),
                future_t.numpy(),
                args.chunks,
            )
        )
        if position == 1 or position % 8 == 0 or position == len(paths):
            print(json.dumps({"exported": position, "total": len(paths)}), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        candidate=np.stack(predictions),
        path=np.asarray(paths),
    )
    manifest = {
        "format": "strict-track2-v202-candidate-only-cache-v1",
        "baseline_cache": str(args.baseline_cache.resolve()),
        "candidate": str(args.candidate.resolve()),
        "output": str(args.output.resolve()),
        "sequence_count": len(paths),
        "chunks": args.chunks,
        "path_alignment_exact": True,
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
