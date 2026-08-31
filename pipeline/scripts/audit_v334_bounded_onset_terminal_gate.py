#!/usr/bin/env python3
"""Run the frozen v333 causal audit against v334 bounded onset repair."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import audit_v333_hybrid_onset_terminal_gate as inherited
from wam_pipeline.v334_bounded_onset_terminal_runtime import Track2V334BoundedOnsetTerminal


def main() -> int:
    inherited.Track2V333HybridOnsetTerminal = Track2V334BoundedOnsetTerminal
    inherited.main()
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    report = json.loads(output.read_text())
    report["format"] = "strict-track2-v334-bounded-onset-terminal-gate-v1"
    report["candidate"] = "track2-v334-bounded-onset-terminal-v333-v326-v325-v317"
    report["inherited_audit"] = "strict-track2-v333-hybrid-onset-terminal-gate-v1"
    report["target_semantics"] = (
        "same-episode onset only for terminal-ineligible phase offsets 0..2; "
        "all later offsets use the exact v326 terminal selector"
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"v334_passed": report["passed"], "checks": report["checks"]}, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
