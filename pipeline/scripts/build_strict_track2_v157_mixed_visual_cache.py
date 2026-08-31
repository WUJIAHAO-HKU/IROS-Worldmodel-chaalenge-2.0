#!/usr/bin/env python3
"""Build a balanced official/synthetic visual cache from the frozen V15.7 API."""

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


def evenly(values: list, count: int) -> list:
    if len(values) < count:
        raise RuntimeError(f"cannot select {count} values from {len(values)}")
    return [values[int(index)] for index in np.linspace(0, len(values) - 1, count, dtype=int)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--synthetic-cache", type=Path, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8001")
    parser.add_argument("--token", default="local-dev-token")
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise SystemExit("refusing to overwrite mixed V15.7 cache or manifest")
    prereg = json.loads(args.preregistration.read_text())
    if prereg.get("format") != "strict-track2-v157-mixed-visual-cache-preregistration-v1":
        raise SystemExit("unexpected mixed-cache preregistration")
    if sha256(Path(__file__).resolve()) != prereg["builder"]["script_sha256"]:
        raise SystemExit("mixed-cache builder differs from preregistration")
    if sha256(args.source_manifest) != prereg["inputs"]["source_manifest_sha256"]:
        raise SystemExit("window source manifest differs from preregistration")
    if sha256(args.synthetic_cache) != prereg["inputs"]["synthetic_cache_sha256"]:
        raise SystemExit("synthetic cache differs from preregistration")
    if args.model_version != prereg["frozen_v157"]["model_version"]:
        raise SystemExit("V15.7 model version differs from preregistration")

    rows = json.loads(args.source_manifest.read_text())
    official_by_arm = {"left": [], "right": []}
    for row in rows:
        if row["split"] == "validation" and row["source"] == "official":
            official_by_arm[row["arm"]].append(row)
    selected_rows = evenly(official_by_arm["left"], 16) + evenly(official_by_arm["right"], 16)
    contexts, histories, futures, targets, paths, starts, arms = [], [], [], [], [], [], []
    for row in selected_rows:
        path = args.windows / row["path"]
        if sha256(path) != row["sha256"]:
            raise SystemExit(f"official validation window hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as values:
            contexts.append(values["context_frames"].copy())
            histories.append(values["history_actions"].copy())
            futures.append(values["future_actions"].copy())
            targets.append(values["target_frames"].copy())
            starts.append(int(values["start"]))
        paths.append(row["path"])
        arms.append(row["arm"] == "right")
    contexts_array = np.stack(contexts)
    histories_array = np.stack(histories)
    futures_array = np.stack(futures)
    client = Track2ServiceClient(args.api_url, args.token, args.model_version, 900.0)
    client.assert_ready()
    official_prediction = client.predict_batch(
        contexts_array,
        histories_array,
        futures_array,
        np.arange(32, dtype=np.int64),
        [None] * 32,
    )

    with np.load(args.synthetic_cache, allow_pickle=False) as values:
        synthetic = {name: values[name].copy() for name in values.files}
    synthetic_indices = []
    for right in (False, True):
        indices = np.flatnonzero(synthetic["arm_right"] == right).tolist()
        synthetic_indices.extend(evenly(indices, 16))
    synthetic_indices = np.asarray(synthetic_indices, dtype=np.int64)
    output_values = {
        "context_last": np.concatenate((contexts_array[:, -1], synthetic["context_last"][synthetic_indices])),
        "target": np.concatenate((np.stack(targets), synthetic["target"][synthetic_indices])),
        "candidate": np.concatenate((official_prediction, synthetic["candidate"][synthetic_indices])),
        "arm_right": np.concatenate((np.asarray(arms, dtype=bool), synthetic["arm_right"][synthetic_indices])),
        "capture_success": np.concatenate((np.ones(32, dtype=bool), synthetic["capture_success"][synthetic_indices])),
        "path": np.concatenate((np.asarray(paths), synthetic["path"][synthetic_indices])),
        "synthetic_seed": np.concatenate((np.full(32, -1, dtype=np.int64), synthetic["synthetic_seed"][synthetic_indices])),
        "start": np.concatenate((np.asarray(starts, dtype=np.int64), synthetic["start"][synthetic_indices])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output_values)
    document = {
        "format": "strict-track2-v157-balanced-mixed-visual-cache-v1",
        "cache": str(args.output.resolve()),
        "cache_sha256": sha256(args.output),
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": sha256(Path(__file__).resolve()),
        "preregistration": str(args.preregistration.resolve()),
        "preregistration_sha256": sha256(args.preregistration),
        "frozen_model_version": args.model_version,
        "samples": 64,
        "counts": {
            "official_left_success": 16,
            "official_right_success": 16,
            "synthetic_left": 16,
            "synthetic_right": 16,
        },
        "official_paths": paths,
        "synthetic_source_indices": synthetic_indices.tolist(),
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": sha256(args.source_manifest),
        "synthetic_cache": str(args.synthetic_cache.resolve()),
        "synthetic_cache_sha256": sha256(args.synthetic_cache),
        "data_guard": "episode-disjoint parent validation only; no development22 or final128 seed",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
