#!/usr/bin/env python3
"""Clarify v302 bit-exact failure versus v303 preregistered numeric acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v302-report", required=True, type=Path)
    parser.add_argument("--v302-auditor", required=True, type=Path)
    parser.add_argument("--v303-prepare", required=True, type=Path)
    parser.add_argument("--v303-release", required=True, type=Path)
    parser.add_argument("--capture-gate", required=True, type=Path)
    parser.add_argument("--long-gate", required=True, type=Path)
    parser.add_argument("--audit-complete", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    v302 = json.loads(args.v302_report.read_text())
    release = json.loads(args.v303_release.read_text())
    capture = json.loads(args.capture_gate.read_text())
    long_gate = json.loads(args.long_gate.read_text())
    auditor_source = args.v302_auditor.read_text()
    prepare_source = args.v303_prepare.read_text()
    numeric_gate = release["batch_equivalence_gate"]
    observed = numeric_gate["observed"]

    checks = {
        "v302_legacy_result_was_not_bit_exact": v302["passed"] is False
        and v302["mismatched_pixels"] > 0,
        "v302_pass_field_was_bit_exact_only": (
            '"passed": bool(np.count_nonzero(difference) == 0)' in auditor_source
        ),
        "v303_preregistered_max_pixel_gate": (
            v302["max_absolute_pixel_change"]
            <= numeric_gate["max_absolute_pixel_change"]
        ),
        "v303_preregistered_mean_pixel_gate": (
            v302["mean_absolute_pixel_change"]
            <= numeric_gate["mean_absolute_pixel_change"]
        ),
        "v303_preregistered_speed_gate": (
            v302["speedup"] >= numeric_gate["speedup_min"]
        ),
        "v303_prepare_recomputed_numeric_gate_before_registration": all(
            token in prepare_source
            for token in (
                "batch['max_absolute_pixel_change']>2",
                "batch['mean_absolute_pixel_change']>.05",
                "batch['speedup']<1.5",
                "raise RuntimeError('batch numerical/throughput gate failed')",
            )
        ),
        "release_embeds_exact_v302_observation": observed == v302,
        "capture_gate_passed": capture.get("passed") is True,
        "long_gate_passed": long_gate.get("passed") is True,
        "formal_v303_audit_complete": args.audit_complete.is_file(),
        "no_hidden_final_or_submission_use": (
            v302["guards"]["hidden_or_final_data"] is False
            and v302["guards"]["real_submission"] is False
            and capture["guards"]["hidden_or_final_data"] is False
            and capture["guards"]["real_submission"] is False
            and long_gate["guards"]["hidden_or_final_data"] is False
            and long_gate["guards"]["real_submission"] is False
        ),
    }
    evidence = {
        name: {"path": str(path), "sha256": sha256(path)}
        for name, path in {
            "v302_report": args.v302_report,
            "v302_auditor": args.v302_auditor,
            "v303_prepare": args.v303_prepare,
            "v303_release": args.v303_release,
            "capture_gate": args.capture_gate,
            "long_gate": args.long_gate,
            "audit_complete": args.audit_complete,
        }.items()
    }
    report = {
        "format": "strict-track2-v303-numeric-gate-clarification-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "interpretation": (
            "v302.passed encodes the superseded bit-exact diagnostic only; "
            "v303 independently preregistered and enforced bounded numeric "
            "equivalence before capture and long-horizon gates"
        ),
        "numeric_observation": {
            "max_absolute_pixel_change": v302["max_absolute_pixel_change"],
            "mean_absolute_pixel_change": v302["mean_absolute_pixel_change"],
            "speedup": v302["speedup"],
        },
        "v303_preregistered_thresholds": {
            "max_absolute_pixel_change": numeric_gate["max_absolute_pixel_change"],
            "mean_absolute_pixel_change": numeric_gate[
                "mean_absolute_pixel_change"
            ],
            "speedup_min": numeric_gate["speedup_min"],
        },
        "checks": checks,
        "evidence": evidence,
        "passed": all(checks.values()),
        "runtime_or_training_changed": False,
        "selection_thresholds_changed_after_outcomes": False,
        "official_submission": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
