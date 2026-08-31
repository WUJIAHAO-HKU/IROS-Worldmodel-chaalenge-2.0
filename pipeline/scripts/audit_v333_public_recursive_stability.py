#!/usr/bin/env python3
"""Run the frozen public recursive gate with v333 as the candidate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import audit_v328_public_recursive_stability as inherited
from wam_pipeline.v333_hybrid_onset_terminal_runtime import Track2V333HybridOnsetTerminal


def main() -> int:
    inherited.Track2V328CoherentPhaseTrajectory = Track2V333HybridOnsetTerminal
    inherited.main()
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    report = json.loads(output.read_text())
    report["format"] = "strict-track2-v333-public-recursive-stability-gate-v1"
    report["candidate"] = "track2-v333-hybrid-onset-terminal-v326-v325-v317"
    report["candidate_alias_in_inherited_audit"] = "v328"
    report["aggregates"]["v333"] = report["aggregates"].pop("v328")
    report["rows"]["v333"] = report["rows"].pop("v328")
    report["v333_over_v326_comparisons"] = report.pop("v328_over_v326_comparisons")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "v333_passed": report["passed"],
        "phase_qualified_counts": report["phase_qualified_counts"],
        "v333_over_v326_comparisons": report["v333_over_v326_comparisons"],
        "checks": report["checks"],
    }, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
