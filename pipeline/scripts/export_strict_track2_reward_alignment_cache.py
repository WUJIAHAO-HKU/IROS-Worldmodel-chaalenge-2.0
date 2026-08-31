#!/usr/bin/env python3
"""Export paired GT/baseline/candidate rollouts for frozen reward-model auditing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_strict_track2_autoregressive_candidate import Windows
from wam_pipeline.v15_gated_runtime import Track2V15GatedRuntime
from wam_pipeline.v15_runtime import Track2V15Runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--episodes-key", default="validation_episodes")
    parser.add_argument("--baseline-release")
    parser.add_argument("--candidate-release", required=True)
    parser.add_argument(
        "--reuse-baseline-cache",
        help="Reuse context/target/baseline arrays from an identically selected cache",
    )
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-windows", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    split = json.loads(Path(args.split_manifest).read_text())
    dataset = Windows(Path(args.windows), split[args.episodes_key])
    indices = list(range(len(dataset)))
    if args.max_windows and args.max_windows < len(indices):
        indices = np.linspace(0, len(dataset) - 1, args.max_windows, dtype=int).tolist()

    if not args.baseline_release and not args.reuse_baseline_cache:
        raise SystemExit("--baseline-release is required without --reuse-baseline-cache")
    baseline = None
    reused = None
    if args.reuse_baseline_cache:
        reused = np.load(args.reuse_baseline_cache, allow_pickle=False)
        expected_paths = [dataset.paths[index].name for index in indices]
        if reused["path"].astype(str).tolist() != expected_paths:
            raise ValueError("reused cache does not contain the identical selected windows")
    else:
        baseline = Track2V15Runtime(args.baseline_release, args.library, args.device)
    candidate = Track2V15GatedRuntime(args.candidate_release, args.library, args.device)
    records: dict[str, list[np.ndarray] | list[str]] = {
        "context_last": [],
        "target": [],
        "baseline": [],
        "candidate": [],
        "arm_right": [],
        "capture_success": [],
        "path": [],
        "synthetic_seed": [],
        "start": [],
    }
    route_records = []
    for position, index in enumerate(indices, 1):
        context, history, future, target, arm_right, capture_success = dataset[index]
        if reused is not None:
            baseline_prediction = reused["baseline"][position - 1].copy()
            if not np.array_equal(reused["context_last"][position - 1], context[-1]):
                raise ValueError(f"reused context mismatch: {dataset.paths[index].name}")
            if not np.array_equal(reused["target"][position - 1], target):
                raise ValueError(f"reused target mismatch: {dataset.paths[index].name}")
        else:
            baseline_prediction = baseline.predict(context, history, future, 0, None)
        candidate_prediction = candidate.predict(context, history, future, 0, None)
        path = dataset.paths[index]
        with np.load(path, allow_pickle=False) as values:
            synthetic_seed = int(values["synthetic_seed"]) if "synthetic_seed" in values else -1
            start = int(values["start"]) if "start" in values else -1
        records["context_last"].append(context[-1])
        records["target"].append(target)
        records["baseline"].append(baseline_prediction)
        records["candidate"].append(candidate_prediction)
        records["arm_right"].append(np.asarray(arm_right, dtype=np.bool_))
        records["capture_success"].append(np.asarray(capture_success, dtype=np.bool_))
        records["path"].append(path.name)
        records["synthetic_seed"].append(np.asarray(synthetic_seed, dtype=np.int64))
        records["start"].append(np.asarray(start, dtype=np.int64))
        route_records.append(
            {
                "path": path.name,
                "route": candidate.gated_autoregressive.last_route,
                "probability_synthetic": candidate.gated_autoregressive.last_probability_synthetic,
            }
        )
        if position == 1 or position % 8 == 0 or position == len(indices):
            print(json.dumps({"exported": position, "total": len(indices)}), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        **{
            key: np.stack(value) if key != "path" else np.asarray(value)
            for key, value in records.items()
        },
    )
    manifest = {
        "format": "strict-track2-reward-alignment-cache-v1",
        "output": str(output.resolve()),
        "window_count": len(indices),
        "frame_count_per_window": 8,
        "split_manifest": str(Path(args.split_manifest).resolve()),
        "episodes_key": args.episodes_key,
        "baseline_release": str(Path(args.baseline_release).resolve()) if args.baseline_release else None,
        "reused_baseline_cache": str(Path(args.reuse_baseline_cache).resolve()) if args.reuse_baseline_cache else None,
        "candidate_release": str(Path(args.candidate_release).resolve()),
        "candidate_routes": {
            "candidate": sum(row["route"] == "candidate" for row in route_records),
            "baseline": sum(row["route"] == "baseline" for row in route_records),
            "records": route_records,
        },
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if reused is not None:
        reused.close()
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
