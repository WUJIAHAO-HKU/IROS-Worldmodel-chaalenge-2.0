#!/usr/bin/env python3
"""Summarize the strict V15.7 complementary-offload one-update diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


FATAL = re.compile(
    r"CUDA out of memory|OutOfMemoryError|RayTaskError|Traceback \(most recent call last\)",
    re.IGNORECASE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensorboard_scalars(root: Path) -> dict:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {"available": False}
    output = {"available": True, "files": {}}
    for path in sorted(root.rglob("events.out.tfevents.*")):
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
        try:
            accumulator.Reload()
        except Exception as exc:
            output["files"][str(path)] = {"error": str(exc)}
            continue
        values = {}
        for tag in accumulator.Tags().get("scalars", []):
            events = accumulator.Scalars(tag)
            if events:
                values[tag] = {
                    "count": len(events),
                    "last_step": int(events[-1].step),
                    "last_value": float(events[-1].value),
                    "min": float(min(event.value for event in events)),
                    "max": float(max(event.value for event in events)),
                }
        output["files"][str(path)] = values
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--bridge-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    launcher = args.run / "launcher.log"
    gpu_trace = args.run / "audit" / "gpu_process_memory_500ms.csv"
    checkpoint = (
        args.run
        / "wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05"
        / "checkpoints"
        / "global_step_1"
        / "actor"
    )
    launcher_text = launcher.read_text(errors="replace")
    bridge_text = args.bridge_log.read_text(errors="replace")
    chunks = bridge_text.count("POST /chunk_step")
    resets = bridge_text.count("POST /reset")
    fatal_lines = [line for line in launcher_text.splitlines() if FATAL.search(line)]

    per_process: dict[tuple[str, str], int] = defaultdict(int)
    aggregate: dict[str, int] = defaultdict(int)
    with gpu_trace.open(newline="") as stream:
        for row in csv.reader(stream, skipinitialspace=True):
            if len(row) < 4:
                continue
            timestamp, pid, memory, name = row[:4]
            match = re.search(r"(\d+)", memory)
            if match is None:
                continue
            mib = int(match.group(1))
            per_process[(pid, name)] = max(per_process[(pid, name)], mib)
            aggregate[timestamp] += mib
    peak_timestamp = max(aggregate, key=aggregate.get) if aggregate else None
    checkpoint_files = sorted(path for path in checkpoint.rglob("*") if path.is_file()) if checkpoint.is_dir() else []
    checkpoint_bytes = sum(path.stat().st_size for path in checkpoint_files)
    checks = {
        "world_model_chunks_exact_200": chunks == 200,
        "rollout_resets_exact_8": resets == 8,
        "fatal_error_absent": not fatal_lines,
        "gpu_completion_marker": (args.run / "audit" / "gpu_after_completion.csv").is_file(),
        "checkpoint_global_step_1_nonempty": bool(checkpoint_files),
        "checkpoint_log_marker": "Saving checkpoint at step 1" in launcher_text,
    }
    report = {
        "format": "strict-track2-v157-complementary-offload-result-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "classification": "resource and one-update policy-signal diagnostic; nonformal",
        "run": str(args.run.resolve()),
        "bridge_log": str(args.bridge_log.resolve()),
        "counts": {"world_model_chunks": chunks, "resets": resets, "fatal_lines": len(fatal_lines)},
        "fatal_excerpt": fatal_lines[:20],
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "files": len(checkpoint_files),
            "bytes": checkpoint_bytes,
            "relative_files": [str(path.relative_to(checkpoint)) for path in checkpoint_files],
        },
        "sampled_gpu": {
            "per_process_peak_mib": [
                {"pid": int(pid), "name": name, "peak_mib": peak}
                for (pid, name), peak in sorted(per_process.items(), key=lambda item: item[1], reverse=True)
            ],
            "aggregate_process_peak_mib": aggregate.get(peak_timestamp, 0) if peak_timestamp else 0,
            "aggregate_process_peak_timestamp": peak_timestamp,
            "trace": str(gpu_trace.resolve()),
            "trace_sha256": sha256(gpu_trace),
        },
        "tensorboard": tensorboard_scalars(args.run / "tensorboard"),
        "evidence": {
            "launcher_sha256": sha256(launcher),
            "bridge_log_sha256": sha256(args.bridge_log),
        },
        "interpretation_guard": "Passing proves one exact update is executable and checkpointed; it is not a full-budget or real RoboTwin Track 2 success result.",
    }
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
