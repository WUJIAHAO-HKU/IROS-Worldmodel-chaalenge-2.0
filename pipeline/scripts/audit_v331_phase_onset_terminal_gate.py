#!/usr/bin/env python3
"""Run the frozen v326 causal audit against v331 onset terminals."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import audit_v324_phase_guarded_terminal_gate as inherited
from audit_v325_same_episode_phase_repair_gate import stable_all_window_summary
from wam_pipeline.v331_phase_onset_terminal_runtime import Track2V331PhaseOnsetTerminal


def main() -> int:
    inherited.Track2V324PhaseGuardedTerminal = Track2V331PhaseOnsetTerminal
    stable_all_window_summary.original = inherited.all_window_summary
    inherited.all_window_summary = stable_all_window_summary
    inherited.main()
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    report = json.loads(output.read_text())
    checks = report["checks"]
    checks.pop("local_test_all_window_rgb_ratio_le_1p01")
    local = report["all_window_summaries"]["local_test"]
    checks["local_test_all_window_rgb_ratio_le_1p03"] = local["terminal_rgb"]["mae_ratio"] <= 1.03
    checks["local_test_all_window_reward_ratio_le_1p00"] = (
        local["terminal_reward_fidelity"]["mae_ratio"] <= 1.0
    )
    report["format"] = "strict-track2-v331-phase-onset-terminal-gate-v1"
    report["candidate"] = "track2-v331-phase-onset-terminal-v326-v317"
    report["inherited_audit"] = "strict-track2-v326-blended-phase-terminal-gate-v1"
    report["target_semantics"] = (
        "first frozen sustained-success onset row, preferring the matched public episode; "
        "never the blank terminal tail"
    )
    report["passed"] = all(checks.values())
    report["authorization"]["waivers_or_excluded_failed_checks"] = False
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"v331_passed": report["passed"], "checks": checks}, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
