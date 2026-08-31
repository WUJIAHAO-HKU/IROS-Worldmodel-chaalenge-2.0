#!/usr/bin/env python3
"""Episode-disjoint audit for the v292 terminal-preserving mirror model."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from wam_pipeline.v271_endpoint_calibrated_terminal_runtime import (
    Track2V271EndpointCalibratedTerminal,
)
from wam_pipeline.v292_terminal_preserving_mirror_runtime import (
    Track2V292TerminalPreservingMirror,
)
from wam_pipeline.v295_terminal_frame_preserving_mirror_runtime import (
    Track2V295TerminalFramePreservingMirror,
)


def evenly_spaced(paths: list[Path], count: int) -> list[Path]:
    if count >= len(paths):
        return paths
    positions = np.linspace(0, len(paths) - 1, count).round().astype(int)
    return [paths[int(position)] for position in positions]


def moving_mae(prediction: np.ndarray, target: np.ndarray, anchor: np.ndarray) -> float:
    motion = np.abs(target.astype(np.int16) - anchor.astype(np.int16)[None]).mean(axis=-1)
    mask = motion >= 4.0
    error = np.abs(prediction.astype(np.int16) - target.astype(np.int16)).mean(axis=-1)
    return float(error[mask].mean()) if mask.any() else float(error.mean())


def summarize(rows: list[dict]) -> dict:
    direct = np.asarray([row["direct_mae"] for row in rows])
    candidate = np.asarray([row["candidate_mae"] for row in rows])
    direct_moving = np.asarray([row["direct_moving_mae"] for row in rows])
    candidate_moving = np.asarray([row["candidate_moving_mae"] for row in rows])
    return {
        "windows": len(rows),
        "mirror_applied": int(sum(row["mirror_applied"] for row in rows)),
        "direct_mae": float(direct.mean()),
        "candidate_mae": float(candidate.mean()),
        "relative_mae_change": float(candidate.mean() / direct.mean() - 1.0),
        "candidate_better_fraction": float(np.mean(candidate < direct)),
        "direct_moving_mae": float(direct_moving.mean()),
        "candidate_moving_mae": float(candidate_moving.mean()),
        "relative_moving_mae_change": float(
            candidate_moving.mean() / direct_moving.mean() - 1.0
        ),
        "per_horizon_direct_mae": np.mean(
            [row["per_horizon_direct_mae"] for row in rows], axis=0
        ).tolist(),
        "per_horizon_candidate_mae": np.mean(
            [row["per_horizon_candidate_mae"] for row in rows], axis=0
        ).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--library-index", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-split", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--candidate", choices=("v292", "v295"), default="v292")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    split = json.loads(args.split_manifest.read_text())
    right_episodes = {"validation": [7, 18], "local_test": [6, 22]}
    direct_runtime = Track2V271EndpointCalibratedTerminal(
        args.checkpoint_dir, args.library_index, args.device
    )
    candidate_class = (
        Track2V292TerminalPreservingMirror
        if args.candidate == "v292"
        else Track2V295TerminalFramePreservingMirror
    )
    candidate_runtime = candidate_class(
        args.checkpoint_dir, args.library_index, args.device
    )
    all_rows: dict[str, list[dict]] = {}
    for split_name, episodes in right_episodes.items():
        manifest_key = f"{split_name}_episodes"
        if not set(episodes).issubset(set(split[manifest_key])):
            raise RuntimeError(f"episode declaration conflicts with {manifest_key}")
        paths = [
            path
            for episode in episodes
            for path in sorted(args.windows.glob(f"episode{episode}_*.npz"))
        ]
        selected = evenly_spaced(paths, args.samples_per_split)
        rows: list[dict] = []
        for index, path in enumerate(selected):
            with np.load(path, allow_pickle=False) as data:
                context = np.asarray(data["context_frames"], dtype=np.uint8)
                history = np.asarray(data["history_actions"], dtype=np.float32)
                future = np.asarray(data["future_actions"], dtype=np.float32)
                target = np.asarray(data["target_frames"], dtype=np.uint8)
            seed = int.from_bytes(
                hashlib.sha256(path.name.encode()).digest()[:8], "little"
            ) % (2**31)
            direct = direct_runtime.predict(context, history, future, seed, None)
            candidate = candidate_runtime.predict(context, history, future, seed, None)
            direct_error = np.abs(direct.astype(np.int16) - target.astype(np.int16))
            candidate_error = np.abs(candidate.astype(np.int16) - target.astype(np.int16))
            rows.append(
                {
                    "window": path.name,
                    "seed": seed,
                    "mirror_applied": bool(candidate_runtime.last_mirror_applied),
                    "right_gripper_mean": float(future[:, 13].mean()),
                    "direct_mae": float(direct_error.mean()),
                    "candidate_mae": float(candidate_error.mean()),
                    "direct_moving_mae": moving_mae(direct, target, context[-1]),
                    "candidate_moving_mae": moving_mae(candidate, target, context[-1]),
                    "per_horizon_direct_mae": direct_error.mean((1, 2, 3)).tolist(),
                    "per_horizon_candidate_mae": candidate_error.mean((1, 2, 3)).tolist(),
                }
            )
            if (index + 1) % 8 == 0 or index + 1 == len(selected):
                print(f"V293_PROGRESS {split_name} {index + 1}/{len(selected)}", flush=True)
        all_rows[split_name] = rows

    report = {
        "format": "strict-track2-v293-v292-terminal-preserving-equivariance-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent": "track2-v271-endpoint-calibrated-terminal",
        "candidate": args.candidate,
        "summaries": {name: summarize(rows) for name, rows in all_rows.items()},
        "rows": all_rows,
        "guards": {
            "participant_component": "world-model RGB predictor only",
            "episode_disjoint_public_windows_only": True,
            "policy_modified": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summaries"], indent=2))


if __name__ == "__main__":
    main()
