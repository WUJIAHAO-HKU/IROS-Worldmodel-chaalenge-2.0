#!/usr/bin/env python3
"""Select a public-validation terminal protection threshold for mirror dynamics."""
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
from wam_pipeline.v290_right_closed_mirror_runtime import use_right_closed_mirror


THRESHOLDS = [0.25, 0.50, 0.75, 0.90, 0.99, 1.00, 1.01]


def summarize(rows: list[dict], threshold: float) -> dict:
    direct = np.asarray([row["direct_mae"] for row in rows])
    mirror = np.asarray([row["mirror_mae"] for row in rows])
    direct_moving = np.asarray([row["direct_moving_mae"] for row in rows])
    mirror_moving = np.asarray([row["mirror_moving_mae"] for row in rows])
    use_mirror = np.asarray(
        [
            row["mirror_eligible"] and row["terminal_alpha"] < threshold
            for row in rows
        ],
        dtype=bool,
    )
    candidate = np.where(use_mirror, mirror, direct)
    candidate_moving = np.where(use_mirror, mirror_moving, direct_moving)
    return {
        "threshold": threshold,
        "mirrored": int(use_mirror.sum()),
        "terminal_protected": int(
            sum(
                row["mirror_eligible"] and row["terminal_alpha"] >= threshold
                for row in rows
            )
        ),
        "mae": float(candidate.mean()),
        "relative_mae_change": float(candidate.mean() / direct.mean() - 1.0),
        "moving_mae": float(candidate_moving.mean()),
        "relative_moving_mae_change": float(
            candidate_moving.mean() / direct_moving.mean() - 1.0
        ),
        "better_fraction": float(np.mean(candidate < direct)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--library-index", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--mirror-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    mirror_report = json.loads(args.mirror_report.read_text())
    runtime = Track2V271EndpointCalibratedTerminal(
        args.checkpoint_dir, args.library_index, args.device
    )
    all_rows: dict[str, list[dict]] = {}
    for split_name, source_rows in mirror_report["rows"].items():
        rows: list[dict] = []
        for index, source in enumerate(source_rows):
            path = args.windows / source["window"]
            with np.load(path, allow_pickle=False) as data:
                context = np.asarray(data["context_frames"], dtype=np.uint8)
                history = np.asarray(data["history_actions"], dtype=np.float32)
                future = np.asarray(data["future_actions"], dtype=np.float32)
            seed = int.from_bytes(
                hashlib.sha256(path.name.encode()).digest()[:8], "little"
            ) % (2**31)
            runtime.predict(context, history, future, seed, None)
            alpha = float(getattr(runtime, "last_progressive_alpha", 0.0) or 0.0)
            rows.append(
                {
                    **source,
                    "mirror_eligible": bool(use_right_closed_mirror(history, future)),
                    "terminal_alpha": alpha,
                    "delta_ratio": (
                        None
                        if getattr(runtime, "last_delta_ratio", None) is None
                        else float(runtime.last_delta_ratio)
                    ),
                    "endpoint_quality": (
                        None
                        if getattr(runtime, "last_endpoint_quality", None) is None
                        else float(runtime.last_endpoint_quality)
                    ),
                }
            )
            if (index + 1) % 16 == 0:
                print(f"V294_PROGRESS {split_name} {index + 1}/{len(source_rows)}", flush=True)
        all_rows[split_name] = rows

    sweeps = {
        split_name: [summarize(rows, threshold) for threshold in THRESHOLDS]
        for split_name, rows in all_rows.items()
    }
    # Selection uses only validation. Local-test is a one-time confirmation.
    eligible = [
        result
        for result in sweeps["validation"]
        if result["relative_mae_change"] < 0
        and result["relative_moving_mae_change"] < 0
    ]
    selected = min(
        eligible,
        key=lambda result: result["mae"] + result["moving_mae"],
    )["threshold"]
    report = {
        "format": "strict-track2-v294-terminal-confidence-gate-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_split": "validation",
        "selected_threshold": selected,
        "sweeps": sweeps,
        "selected_results": {
            name: next(row for row in values if row["threshold"] == selected)
            for name, values in sweeps.items()
        },
        "rows": all_rows,
        "guards": {
            "public_episode_disjoint_windows_only": True,
            "policy_modified": False,
            "reward_used_for_threshold_selection": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected_threshold": selected, "sweeps": sweeps}, indent=2))


if __name__ == "__main__":
    main()
