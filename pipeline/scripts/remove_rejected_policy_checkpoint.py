#!/usr/bin/env python3
"""Remove an exactly targeted, publicly rejected, reconstructible checkpoint."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import zipfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = (
        args.run
        / "wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05"
        / "checkpoints"
        / "global_step_1"
        / "actor"
        / "model_state_dict"
        / "full_weights.pt"
    )
    gate_path = args.run / "audit" / args.gate
    marker_path = args.run / "audit" / args.marker
    gate = json.loads(gate_path.read_text())
    if gate.get("accepted") is not False or not marker_path.is_file():
        raise RuntimeError("checkpoint has not been formally rejected")
    resolved_run = args.run.resolve()
    resolved_checkpoint = checkpoint.resolve()
    if resolved_run not in resolved_checkpoint.parents:
        raise RuntimeError("checkpoint escaped the exact run directory")
    if not checkpoint.is_file() or not zipfile.is_zipfile(checkpoint):
        raise RuntimeError("expected complete PyTorch checkpoint")
    report = {
        "format": "strict-track2-rejected-policy-checkpoint-removal-v1",
        "removed_at": datetime.now(timezone.utc).isoformat(),
        "run": str(resolved_run),
        "checkpoint": str(resolved_checkpoint),
        "checkpoint_size": checkpoint.stat().st_size,
        "reason": "public gate rejected; checkpoint is reconstructible from retained preregistration, source, dataset hashes, and logs",
        "gate": str(gate_path),
        "gate_counts": gate.get("counts"),
        "logs_and_audits_retained": True,
        "recoverable_by_rerun": True,
        "reserved_final128_access": False,
        "real_competition_submission": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    checkpoint.unlink()
    if checkpoint.exists():
        raise RuntimeError("checkpoint removal failed")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
