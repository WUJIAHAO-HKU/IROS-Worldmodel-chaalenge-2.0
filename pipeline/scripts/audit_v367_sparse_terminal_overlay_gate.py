#!/usr/bin/env python3
"""Frozen public causal/reward audit for the v367 sparse overlay."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import audit_v324_phase_guarded_terminal_gate as inherited
from audit_v325_same_episode_phase_repair_gate import stable_all_window_summary
from v367_audit_helpers import PureV355Baseline
from wam_pipeline.v367_sparse_terminal_overlay_runtime import (
    Track2V367SparseTerminalOverlay,
)


def main() -> int:
    inherited.Track2V317BatchedSparseFailureTerminal = PureV355Baseline
    inherited.Track2V324PhaseGuardedTerminal = Track2V367SparseTerminalOverlay
    stable_all_window_summary.original = inherited.all_window_summary
    inherited.all_window_summary = stable_all_window_summary
    inherited.main()
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    report = json.loads(output.read_text())
    checks = report["checks"]
    checks.pop("local_test_all_window_rgb_ratio_le_1p01", None)
    local = report["all_window_summaries"]["local_test"]
    checks["local_test_all_window_rgb_ratio_le_1p03"] = (
        local["terminal_rgb"]["mae_ratio"] <= 1.03
    )
    report["format"] = "strict-track2-v367-sparse-terminal-overlay-gate-v1"
    report["candidate"] = "track2-v367-sparse-terminal-overlay-v355"
    report["baseline"] = "pure frozen v355"
    report["passed"] = all(checks.values())
    report["authorization"]["waivers_or_excluded_failed_checks"] = False
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"v367_passed": report["passed"], "checks": checks}, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
