#!/usr/bin/env python3
"""Audited removal of one exact, formally rejected, reconstructible checkpoint."""

from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint-relative", required=True, type=Path)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    runs_root = args.runs_root.resolve(strict=True)
    run = (runs_root / args.run).resolve(strict=True)
    if run.parent != runs_root:
        raise RuntimeError("run is not an immediate child of the declared runs root")
    checkpoint = (run / args.checkpoint_relative).resolve(strict=True)
    if run not in checkpoint.parents:
        raise RuntimeError("checkpoint escaped its exact run directory")
    gate_path = run / "audit" / args.gate
    marker_path = run / "audit" / args.marker
    gate = json.loads(gate_path.read_text())
    rejection_field = next(
        (field for field in ("accepted", "passed") if gate.get(field) is False),
        None,
    )
    if rejection_field is None or not marker_path.is_file():
        raise RuntimeError("checkpoint lacks a formal public rejection")
    if checkpoint.name != "full_weights.pt" or not zipfile.is_zipfile(checkpoint):
        raise RuntimeError("target is not a complete PyTorch full_weights.pt checkpoint")
    if args.output.exists():
        raise FileExistsError(args.output)
    size = checkpoint.stat().st_size
    report = {
        "format": "strict-track2-rejected-policy-checkpoint-removal-v2",
        "removed_at": datetime.now(timezone.utc).isoformat(),
        "runs_root": str(runs_root),
        "run": str(run),
        "checkpoint": str(checkpoint),
        "checkpoint_size": size,
        "gate": str(gate_path),
        "gate_rejection_field": rejection_field,
        "gate_counts": gate.get("counts"),
        "marker": str(marker_path),
        "reason": "formal public gate rejection; fully reconstructible from retained registration, code, hashes, and logs",
        "logs_and_audits_retained": True,
        "recoverable_by_rerun": True,
        "reserved_final128_access": False,
        "real_competition_submission": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    checkpoint.unlink()
    if checkpoint.exists():
        raise RuntimeError("exact checkpoint removal failed")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
