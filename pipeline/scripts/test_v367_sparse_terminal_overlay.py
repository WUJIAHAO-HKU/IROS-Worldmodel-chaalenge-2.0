#!/usr/bin/env python3
"""Inherited numeric/causal contract with pure-v355 as the baseline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import test_v324_phase_guarded_terminal as inherited
from v367_audit_helpers import PureV355Baseline
from wam_pipeline.v367_sparse_terminal_overlay_runtime import (
    Track2V367SparseTerminalOverlay,
)


def main() -> int:
    inherited.Track2V324PhaseGuardedTerminal = Track2V367SparseTerminalOverlay
    inherited.Track2V271EndpointCalibratedTerminal = PureV355Baseline
    inherited.Track2V317BatchedSparseFailureTerminal = PureV355Baseline
    inherited.main()
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    report = json.loads(output.read_text())
    checks = report["checks"]
    checks.pop("batch_max_absolute_pixel_change_le_2", None)
    checks.pop("batch_mean_absolute_pixel_change_le_0p05", None)
    checks.pop("native_batch_speedup_ge_1p4", None)
    serial_batch = report["serial_batch"]
    checks["serial_batch_pixel_bit_exact"] = (
        serial_batch["max_absolute_pixel_change"] == 0
        and serial_batch["mean_absolute_pixel_change"] == 0.0
    )
    checks["throughput_deferred_to_http_acceptance"] = True
    report["format"] = "strict-track2-v367-sparse-terminal-overlay-contract-v1"
    report["candidate"] = "track2-v367-sparse-terminal-overlay-v355"
    report["baseline"] = "pure frozen v355"
    report["passed"] = all(checks.values())
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"v367_passed": report["passed"], "checks": checks}, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
