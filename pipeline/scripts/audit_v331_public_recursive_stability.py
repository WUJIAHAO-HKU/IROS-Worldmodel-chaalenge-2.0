#!/usr/bin/env python3
"""Run the frozen public recursive gate with v331 as the candidate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import audit_v328_public_recursive_stability as inherited
from wam_pipeline.v331_phase_onset_terminal_runtime import Track2V331PhaseOnsetTerminal


def main() -> int:
    inherited.Track2V328CoherentPhaseTrajectory = Track2V331PhaseOnsetTerminal
    inherited.main()
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    report = json.loads(output.read_text())
    report["format"] = "strict-track2-v331-public-recursive-stability-gate-v1"
    report["candidate"] = "track2-v331-phase-onset-terminal-v326-v317"
    report["candidate_alias_in_inherited_audit"] = "v328"
    report["aggregates"]["v331"] = report["aggregates"].pop("v328")
    report["rows"]["v331"] = report["rows"].pop("v328")
    report["v331_over_v326_comparisons"] = report.pop("v328_over_v326_comparisons")
    report["replay_rule"]["teacher_phase_subset"] = (
        "frozen v331/v326 action, phase, baseline-alpha, and failure gates on public ground-truth context"
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "v331_passed": report["passed"],
        "phase_qualified_counts": report["phase_qualified_counts"],
        "v331_over_v326_comparisons": report["v331_over_v326_comparisons"],
        "checks": report["checks"],
    }, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
