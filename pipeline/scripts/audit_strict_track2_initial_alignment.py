#!/usr/bin/env python3
"""Prove that an official bridge reset obeys the Track 2 transition contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--reference-window", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with np.load(args.audit, allow_pickle=False) as values:
        context = values["context_frames"].copy()
        history = values["history_actions"].copy()
    with np.load(args.reference_window, allow_pickle=False) as values:
        reference_context = values["context_frames"].copy()
        reference_history = values["history_actions"].copy()
        reference_start = int(values["start"])

    batch = int(context.shape[0])
    if context.shape != (batch, 5, 256, 256, 3):
        raise ValueError(f"unexpected audited context shape {context.shape}")
    if history.shape != (batch, 4, 14):
        raise ValueError(f"unexpected audited history shape {history.shape}")
    if reference_context.shape != context.shape[1:]:
        raise ValueError("reference context shape does not match one audited sample")
    if reference_history.shape != history.shape[1:]:
        raise ValueError("reference history shape does not match one audited sample")

    rows = []
    for index in range(batch):
        context_diff = np.abs(
            context[index].astype(np.float32) - reference_context.astype(np.float32)
        )
        history_diff = np.abs(history[index] - reference_history)
        rows.append(
            {
                "batch_index": index,
                "context_exact": bool(np.array_equal(context[index], reference_context)),
                "history_exact": bool(np.array_equal(history[index], reference_history)),
                "context_mean_absolute_difference": float(context_diff.mean()),
                "history_mean_absolute_difference": float(history_diff.mean()),
                "context_sha256": sha256(context[index]),
                "history_sha256": sha256(history[index]),
            }
        )

    report = {
        "format": "strict-track2-initial-alignment-audit-v1",
        "audit": str(args.audit.resolve()),
        "reference_window": str(args.reference_window.resolve()),
        "reference_start": reference_start,
        "reference_context_sha256": sha256(reference_context),
        "reference_history_sha256": sha256(reference_history),
        "contract": {
            "context_images": "o0..o4",
            "history_actions": "a0..a3",
            "pi05_absolute_state": "a4",
        },
        "all_contexts_exact": all(row["context_exact"] for row in rows),
        "all_histories_exact": all(row["history_exact"] for row in rows),
        "passed": all(
            row["context_exact"] and row["history_exact"] for row in rows
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise RuntimeError(f"initial Track 2 alignment failed: {report}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
