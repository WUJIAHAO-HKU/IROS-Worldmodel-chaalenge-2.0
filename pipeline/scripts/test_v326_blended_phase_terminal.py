#!/usr/bin/env python3
"""Run the frozen v324 numeric contract against v326."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import test_v324_phase_guarded_terminal as inherited
from wam_pipeline.v326_blended_phase_terminal_runtime import (
    Track2V326BlendedPhaseTerminal,
)


def main() -> int:
    inherited.Track2V324PhaseGuardedTerminal = Track2V326BlendedPhaseTerminal
    result = inherited.main()
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    if output.is_file():
        report = json.loads(output.read_text())
        report["format"] = "strict-track2-v326-blended-phase-terminal-contract-v1"
        report["candidate"] = "track2-v326-blended-phase-terminal-v317"
        report["inherited_contract"] = "strict-track2-v324-phase-guarded-terminal-contract-v1"
        output.write_text(json.dumps(report, indent=2) + "\n")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
