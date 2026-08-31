#!/usr/bin/env python3
"""Record a fixed half-contraction calibration on public training demos only."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from wam_pipeline.v245_clean_progressive_successor_runtime import (
    Track2V245CleanProgressiveSuccessor,
)
from wam_pipeline.v407_one_chunk_progressive_runtime import (
    Track2V407OneChunkProgressive,
)


FORMAT = "strict-track2-v409-public-train-half-contraction-calibration-v1"
CONTRACTION_SCALE = 0.5


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantiles(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--library-index", required=True, type=Path)
    parser.add_argument("--windows", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    split = json.loads(args.split.read_text())
    train = {int(value) for value in split["train_episodes"]}
    validation = {int(value) for value in split["validation_episodes"]}
    right_train = sorted(
        episode
        for episode in train
        if split["arm_by_episode"][str(episode)] == "right"
    )
    if len(train) != 40 or len(validation) != 10 or len(right_train) != 15:
        raise RuntimeError("public split drift")
    if train & validation:
        raise RuntimeError("train/validation overlap")

    runtime = Track2V407OneChunkProgressive(
        args.checkpoint_dir, args.library_index, args.device
    )
    raw_alpha: list[float] = []
    effective_alpha: list[float] = []
    counts = {
        "windows": 0,
        "post_grasp": 0,
        "signature_rejected": 0,
        "progressive_eligible": 0,
    }
    for episode in right_train:
        for path in sorted(args.windows.glob(f"episode{episode}_*.npz")):
            with np.load(path, allow_pickle=False) as values:
                context = values["context_frames"].astype(np.uint8)
                history = values["history_actions"].astype(np.float32)
                future = values["future_actions"].astype(np.float32)
            counts["windows"] += 1
            if not runtime._post_grasp(history, future):
                continue
            counts["post_grasp"] += 1
            probability = float(runtime._probability(history, future))
            if runtime._signature(history, future, probability) is not None:
                counts["signature_rejected"] += 1
                continue
            base, distance = runtime._nearest_clean(context, history, future)
            alpha = float(
                Track2V245CleanProgressiveSuccessor._alpha(
                    runtime, history, future, base, distance
                )
            )
            counts["progressive_eligible"] += 1
            raw_alpha.append(alpha)
            effective_alpha.append(alpha * CONTRACTION_SCALE)

    if counts["windows"] < 1000 or counts["progressive_eligible"] < 100:
        raise RuntimeError(f"insufficient public-train calibration coverage: {counts}")
    report = {
        "format": FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "method": "fixed recursive convex contraction; no parameter sweep",
            "contraction_scale": CONTRACTION_SCALE,
            "effective_alpha_hard_max": CONTRACTION_SCALE,
            "rationale": "retain at least one-half parent contribution on every progressive frame",
        },
        "coverage": {
            "train_episode_count": len(train),
            "right_train_episodes": right_train,
            **counts,
        },
        "raw_progressive_alpha": quantiles(raw_alpha),
        "contracted_progressive_alpha": quantiles(effective_alpha),
        "evidence_sha256": {
            "split": sha256(args.split),
            "library_index": sha256(args.library_index),
            "v407_manifest": sha256(
                args.checkpoint_dir / "one_chunk_progressive_manifest.json"
            ),
        },
        "guards": {
            "public_train_windows_only": True,
            "validation_windows_read": False,
            "reward_or_outcomes_read": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
