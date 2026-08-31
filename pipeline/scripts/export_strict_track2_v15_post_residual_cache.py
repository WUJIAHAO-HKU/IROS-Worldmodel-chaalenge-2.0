#!/usr/bin/env python3
"""Cache complete frozen-V15 outputs for bounded post-model residual training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from evaluate_strict_track2_autoregressive_candidate import Windows
from wam_pipeline.v15_runtime import Track2V15Runtime


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--episodes-key", default="train_episodes")
    parser.add_argument("--baseline-release", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-start", type=int)
    parser.add_argument("--max-start", type=int)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    split_path = Path(args.split_manifest)
    split = json.loads(split_path.read_text())
    dataset = Windows(
        Path(args.windows), split[args.episodes_key],
        min_start=args.min_start, max_start=args.max_start,
    )
    indices = list(range(len(dataset)))
    if args.max_windows and args.max_windows < len(indices):
        indices = np.linspace(0, len(indices) - 1, args.max_windows, dtype=int).tolist()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"completed cache already exists: {manifest_path}")

    model = Track2V15Runtime(args.baseline_release, args.library, args.device)
    records = []
    counts = {"official": 0, "synthetic": 0, "left": 0, "right": 0}
    for position, index in enumerate(indices, 1):
        context, history, future, target, arm_right, capture_success = dataset[index]
        source_path = dataset.paths[index]
        destination = output / source_path.name
        with np.load(source_path, allow_pickle=False) as source:
            synthetic_seed = int(source["synthetic_seed"]) if "synthetic_seed" in source.files else -1
            start = int(source["start"]) if "start" in source.files else dataset.starts[index]
            instruction = str(source["task_instruction"]) if "task_instruction" in source.files else ""
        if destination.exists():
            with np.load(destination, allow_pickle=False) as cached:
                if str(cached["source_path"]) != str(source_path.resolve()):
                    raise RuntimeError(f"cache collision: {destination}")
        else:
            prediction = model.predict(context, history, future, 0, instruction or None)
            temporary = destination.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    context_last=context[-1],
                    history_actions=history.astype(np.float32),
                    future_actions=future.astype(np.float32),
                    target=target,
                    baseline=prediction,
                    arm_right=np.asarray(arm_right, dtype=np.bool_),
                    capture_success=np.asarray(capture_success, dtype=np.bool_),
                    synthetic_seed=np.asarray(synthetic_seed, dtype=np.int64),
                    start=np.asarray(start, dtype=np.int64),
                    instruction=np.asarray(instruction),
                    source_path=np.asarray(str(source_path.resolve())),
                )
            os.replace(temporary, destination)
        source = "synthetic" if synthetic_seed >= 0 else "official"
        arm = "right" if arm_right else "left"
        counts[source] += 1
        counts[arm] += 1
        records.append({
            "path": destination.name,
            "source_path": str(source_path.resolve()),
            "source": source,
            "arm": arm,
            "capture_success": bool(capture_success),
            "start": start,
            "sha256": sha256(destination),
        })
        if position == 1 or position % 8 == 0 or position == len(indices):
            print(json.dumps({"exported": position, "total": len(indices), "counts": counts}), flush=True)

    manifest = {
        "format": "strict-track2-frozen-v15-post-residual-cache-v1",
        "windows": str(Path(args.windows).resolve()),
        "split_manifest": str(split_path.resolve()),
        "split_manifest_sha256": sha256(split_path),
        "episodes_key": args.episodes_key,
        "min_start": args.min_start,
        "max_start": args.max_start,
        "baseline_release": str(Path(args.baseline_release).resolve()),
        "library": str(Path(args.library).resolve()),
        "count": len(records),
        "counts": counts,
        "records": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"complete": str(manifest_path.resolve()), "count": len(records), "counts": counts}))


if __name__ == "__main__":
    main()
