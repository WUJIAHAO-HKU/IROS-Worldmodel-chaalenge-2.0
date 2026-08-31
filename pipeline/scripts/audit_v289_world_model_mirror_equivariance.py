#!/usr/bin/env python3
"""Compare direct v271 WM predictions with a compliant mirror-equivariant wrapper."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from wam_pipeline.v271_endpoint_calibrated_terminal_runtime import (
    Track2V271EndpointCalibratedTerminal,
)


MIRROR_SIGN = np.asarray([-1, 1, 1, 1, -1, -1, 1], dtype=np.float32)


def mirror_actions(actions: np.ndarray) -> np.ndarray:
    result = np.empty_like(actions, dtype=np.float32)
    result[..., :7] = actions[..., 7:14] * MIRROR_SIGN
    result[..., 7:14] = actions[..., :7] * MIRROR_SIGN
    return result


def mirror_prompt(prompt: str | None) -> str | None:
    if prompt is None:
        return None
    value = re.sub(r"\bleft\b", "__track2_right__", prompt, flags=re.IGNORECASE)
    value = re.sub(r"\bright\b", "left", value, flags=re.IGNORECASE)
    return value.replace("__track2_right__", "right")


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
    mirror = np.asarray([row["mirror_mae"] for row in rows])
    direct_moving = np.asarray([row["direct_moving_mae"] for row in rows])
    mirror_moving = np.asarray([row["mirror_moving_mae"] for row in rows])
    return {
        "windows": len(rows),
        "direct_mae": float(direct.mean()),
        "mirror_mae": float(mirror.mean()),
        "relative_mae_change": float(mirror.mean() / direct.mean() - 1.0),
        "mirror_better_fraction": float(np.mean(mirror < direct)),
        "direct_moving_mae": float(direct_moving.mean()),
        "mirror_moving_mae": float(mirror_moving.mean()),
        "relative_moving_mae_change": float(mirror_moving.mean() / direct_moving.mean() - 1.0),
        "per_horizon_direct_mae": np.mean([row["per_horizon_direct_mae"] for row in rows], axis=0).tolist(),
        "per_horizon_mirror_mae": np.mean([row["per_horizon_mirror_mae"] for row in rows], axis=0).tolist(),
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
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    split = json.loads(args.split_manifest.read_text())
    # Episode-level arm labels are fixed from the public demonstrations and do
    # not use any held-out simulator outcome.
    right_episodes = {"validation": [7, 18], "local_test": [6, 22]}
    runtime = Track2V271EndpointCalibratedTerminal(
        args.checkpoint_dir, args.library_index, args.device
    )
    all_rows = {}
    for split_name, episodes in right_episodes.items():
        manifest_key = f"{split_name}_episodes"
        if not set(episodes).issubset(set(split[manifest_key])):
            raise RuntimeError(f"right episode declaration conflicts with {manifest_key}")
        paths = [
            path
            for episode in episodes
            for path in sorted(args.windows.glob(f"episode{episode}_*.npz"))
        ]
        selected = evenly_spaced(paths, args.samples_per_split)
        rows = []
        for index, path in enumerate(selected):
            data = np.load(path, allow_pickle=False)
            context = np.asarray(data["context_frames"], dtype=np.uint8)
            history = np.asarray(data["history_actions"], dtype=np.float32)
            future = np.asarray(data["future_actions"], dtype=np.float32)
            target = np.asarray(data["target_frames"], dtype=np.uint8)
            seed = int.from_bytes(hashlib.sha256(path.name.encode()).digest()[:8], "little") % (2**31)
            direct = runtime.predict(context, history, future, seed, None)
            mirrored_prediction = runtime.predict(
                np.ascontiguousarray(context[:, :, ::-1, :]),
                mirror_actions(history),
                mirror_actions(future),
                seed,
                mirror_prompt(None),
            )
            mirrored_prediction = np.ascontiguousarray(mirrored_prediction[:, :, ::-1, :])
            direct_error = np.abs(direct.astype(np.int16) - target.astype(np.int16))
            mirror_error = np.abs(mirrored_prediction.astype(np.int16) - target.astype(np.int16))
            rows.append({
                "window": path.name,
                "seed": seed,
                "direct_mae": float(direct_error.mean()),
                "mirror_mae": float(mirror_error.mean()),
                "direct_moving_mae": moving_mae(direct, target, context[-1]),
                "mirror_moving_mae": moving_mae(mirrored_prediction, target, context[-1]),
                "per_horizon_direct_mae": direct_error.mean((1, 2, 3)).tolist(),
                "per_horizon_mirror_mae": mirror_error.mean((1, 2, 3)).tolist(),
            })
            if (index + 1) % 8 == 0 or index + 1 == len(selected):
                print(f"V289_PROGRESS {split_name} {index + 1}/{len(selected)}", flush=True)
        all_rows[split_name] = rows

    report = {
        "format": "strict-track2-v289-world-model-mirror-equivariance-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent": "track2-v271-endpoint-calibrated-terminal",
        "mirror_sign": MIRROR_SIGN.tolist(),
        "summaries": {name: summarize(rows) for name, rows in all_rows.items()},
        "rows": all_rows,
        "rules": {
            "participant_component_modified": "world-model RGB predictor only",
            "actions_rewards_success_or_terminations_returned": False,
            "episode_disjoint_public_windows_only": True,
            "hidden_evaluation_access": False,
            "real_competition_submission": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summaries"], indent=2))
    print(f"V289_AUDIT_COMPLETE {args.output}")


if __name__ == "__main__":
    main()
