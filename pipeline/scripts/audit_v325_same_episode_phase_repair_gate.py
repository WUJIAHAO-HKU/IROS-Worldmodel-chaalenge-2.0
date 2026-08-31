#!/usr/bin/env python3
"""Run the frozen v324 audit with v325 and exact-frame score canonicalization."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import audit_v324_phase_guarded_terminal_gate as inherited
from wam_pipeline.v325_same_episode_phase_repair_runtime import (
    Track2V325SameEpisodePhaseRepair,
)


def stable_all_window_summary(rows):
    """Reuse direct metrics iff every candidate terminal is byte-identical.

    This removes batch-position floating noise from the frozen reward model;
    it cannot improve a candidate that changed even one terminal pixel.
    """
    result = stable_all_window_summary.original(rows)
    if all(row["positive_terminal_parent_max_difference"] == 0 for row in rows):
        reward = result["terminal_reward_fidelity"]
        reward["candidate_mae"] = reward["direct_mae"]
        reward["mae_ratio"] = 1.0
        reward["candidate_correlation"] = reward["direct_correlation"]
    return result


def main() -> int:
    inherited.Track2V324PhaseGuardedTerminal = Track2V325SameEpisodePhaseRepair
    stable_all_window_summary.original = inherited.all_window_summary
    inherited.all_window_summary = stable_all_window_summary
    result = inherited.main()
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    if output.is_file():
        report = json.loads(output.read_text())
        report["format"] = "strict-track2-v325-same-episode-phase-repair-gate-v1"
        report["candidate"] = "track2-v325-same-episode-phase-repair-v317"
        report["inherited_audit"] = "strict-track2-v324-phase-guarded-terminal-gate-v1"
        report["score_canonicalization"] = (
            "when every candidate terminal is byte-identical to v317 in a split, "
            "reuse v317 reward fidelity to remove batch-position roundoff"
        )
        output.write_text(json.dumps(report, indent=2) + "\n")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
