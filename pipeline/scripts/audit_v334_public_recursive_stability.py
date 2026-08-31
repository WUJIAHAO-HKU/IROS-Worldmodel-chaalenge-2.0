#!/usr/bin/env python3
"""Run the frozen public recursive gate with v334 as the candidate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import audit_v333_public_recursive_stability as inherited
from wam_pipeline.v334_bounded_onset_terminal_runtime import Track2V334BoundedOnsetTerminal


def main() -> int:
    inherited.Track2V333HybridOnsetTerminal = Track2V334BoundedOnsetTerminal
    inherited.main()
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    report = json.loads(output.read_text())
    report["format"] = "strict-track2-v334-public-recursive-stability-gate-v1"
    report["candidate"] = "track2-v334-bounded-onset-terminal-v333-v326-v325-v317"
    report["aggregates"]["v334"] = report["aggregates"].pop("v333")
    report["rows"]["v334"] = report["rows"].pop("v333")
    report["v334_over_v326_comparisons"] = report.pop("v333_over_v326_comparisons")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "v334_passed": report["passed"],
        "phase_qualified_counts": report["phase_qualified_counts"],
        "v334_over_v326_comparisons": report["v334_over_v326_comparisons"],
        "checks": report["checks"],
    }, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
