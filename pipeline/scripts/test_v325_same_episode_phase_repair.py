#!/usr/bin/env python3
"""Run the frozen v324 numeric contract against the narrower v325 runtime."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import test_v324_phase_guarded_terminal as inherited
from wam_pipeline.v325_same_episode_phase_repair_runtime import (
    Track2V325SameEpisodePhaseRepair,
)


def main() -> int:
    inherited.Track2V324PhaseGuardedTerminal = Track2V325SameEpisodePhaseRepair
    result = inherited.main()
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    if output.is_file():
        report = json.loads(output.read_text())
        report["format"] = "strict-track2-v325-same-episode-phase-repair-contract-v1"
        report["candidate"] = "track2-v325-same-episode-phase-repair-v317"
        report["inherited_contract"] = "strict-track2-v324-phase-guarded-terminal-contract-v1"
        output.write_text(json.dumps(report, indent=2) + "\n")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
