#!/usr/bin/env python3
"""Explain v217's invalid offline-label first-chunk equality requirement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wam_pipeline.arm_routed_autoregressive_runtime import Track2ArmRoutedAutoregressiveUNet


def parse_name(name: str) -> tuple[int, int]:
    stem = Path(name).stem
    episode, start = stem.split("_")
    return int(episode[7:]), int(start)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--v216", type=Path, required=True)
    parser.add_argument("--success-windows", type=Path, required=True)
    parser.add_argument("--failure-windows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    rows = []
    for tag, windows in (
        ("public_success", args.success_windows),
        ("public_failure", args.failure_windows),
    ):
        with np.load(args.run / f"audit/{tag}_baseline.npz", allow_pickle=False) as values:
            paths = values["path"].astype(str)
            arm_right = values["arm_right"].astype(bool)
            instructions = values["instruction"].astype(str)
        with np.load(args.run / f"audit/{tag}_candidate.npz", allow_pickle=False) as values:
            actual = values["candidate"][:, :8].copy()
        with np.load(args.v216 / f"audit/alpha070_{tag}_candidate.npz", allow_pickle=False) as values:
            offline = values["candidate"][:, :8].copy()
        for index, name in enumerate(paths):
            with np.load(windows / name, allow_pickle=False) as values:
                route = Track2ArmRoutedAutoregressiveUNet.active_arm(
                    values["history_actions"],
                    values["future_actions"],
                    instructions[index],
                )
            difference = np.abs(
                actual[index].astype(np.int16) - offline[index].astype(np.int16)
            )
            label_route = "right" if arm_right[index] else "left"
            rows.append(
                {
                    "set": tag,
                    "path": name,
                    "audit_label_route": label_route,
                    "deployable_request_route": route,
                    "route_matches": route == label_route,
                    "first_chunk_mae": float(difference.mean()),
                    "first_chunk_max_abs": int(difference.max()),
                }
            )
    groups = {}
    for tag in ("public_success", "public_failure"):
        for matched in (True, False):
            subset = [row for row in rows if row["set"] == tag and row["route_matches"] is matched]
            key = f"{tag}_{'matched' if matched else 'mismatched'}"
            groups[key] = {
                "count": len(subset),
                "mean_first_chunk_mae": float(np.mean([row["first_chunk_mae"] for row in subset])) if subset else None,
                "max_first_chunk_mae": float(np.max([row["first_chunk_mae"] for row in subset])) if subset else None,
                "max_abs": int(np.max([row["first_chunk_max_abs"] for row in subset])) if subset else None,
            }
    v217 = json.loads((args.run / "audit/v217_online_recursive_report.json").read_text())
    report = {
        "format": "strict-track2-v217-deployable-route-mismatch-diagnostic-v1",
        "finding": (
            "v216 offline retrieval used arm_right audit metadata, while the real service "
            "correctly routes only from request instruction/actions; bit-exact equality is "
            "therefore invalid when those routes disagree"
        ),
        "groups": groups,
        "v217_substantive_checks": {
            key: value
            for key, value in v217["checks"].items()
            if key != "first_chunk_api_equivalence"
        },
        "v217_success": v217["success"],
        "v217_failure": v217["failure"],
        "rows": rows,
        "hidden_or_final_data": False,
        "real_submission": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
