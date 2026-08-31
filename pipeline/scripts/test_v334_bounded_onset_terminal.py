#!/usr/bin/env python3
"""Run v333 checks plus a frozen late-phase v326 fallback assertion."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import test_v333_hybrid_onset_terminal as inherited
from wam_pipeline.v326_blended_phase_terminal_runtime import Track2V326BlendedPhaseTerminal
from wam_pipeline.v334_bounded_onset_terminal_runtime import (
    ONSET_REPAIR_MAX_OFFSET,
    Track2V334BoundedOnsetTerminal,
)


def main() -> int:
    inherited.Track2V333HybridOnsetTerminal = Track2V334BoundedOnsetTerminal
    inherited_result = inherited.main()
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    if not output.is_file():
        return inherited_result
    report = json.loads(output.read_text())
    checkpoint = Path(sys.argv[sys.argv.index("--checkpoint-dir") + 1])
    library = Path(sys.argv[sys.argv.index("--library-index") + 1])
    action_gate = Path(sys.argv[sys.argv.index("--action-gate") + 1])
    phase_gate = Path(sys.argv[sys.argv.index("--phase-gate") + 1])
    windows = Path(sys.argv[sys.argv.index("--windows") + 1])
    device = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cuda"
    path = windows / "episode6_00120.npz"
    with np.load(path, allow_pickle=False) as values:
        context = values["context_frames"].astype(np.uint8)
        history = values["history_actions"].astype(np.float32)
        future = values["future_actions"].astype(np.float32)
    value_seed = inherited.inherited.seed(path)
    prompt = "adjust bottle"
    candidate_runtime = Track2V334BoundedOnsetTerminal(
        checkpoint, library, device, action_gate, phase_gate
    )
    v326_runtime = Track2V326BlendedPhaseTerminal(
        checkpoint, library, device, action_gate, phase_gate
    )
    base, _ = candidate_runtime._nearest_clean(context, history, future)
    phase_ready, phase_episode, phase_start, phase_onset = candidate_runtime._phase(base)
    candidate = candidate_runtime.predict(context, history, future, value_seed, prompt)
    v326 = v326_runtime.predict(context, history, future, value_seed, prompt)
    late_checks = {
        "late_phase_is_terminal_ineligible": not candidate_runtime.eligible_episode[phase_episode],
        "late_phase_offset_gt_bound": phase_start - phase_onset > ONSET_REPAIR_MAX_OFFSET,
        "late_phase_full_sequence_bit_exact_v326": bool(np.array_equal(candidate, v326)),
    }
    report["format"] = "strict-track2-v334-bounded-onset-terminal-contract-v1"
    report["candidate"] = "track2-v334-bounded-onset-terminal-v333-v326-v325-v317"
    report["inherited_contract"] = "strict-track2-v333-hybrid-onset-terminal-contract-v1"
    report["late_phase_fallback"] = {
        "window": path.name,
        "phase_episode": phase_episode,
        "phase_start": phase_start,
        "phase_onset": phase_onset,
        "phase_offset": phase_start - phase_onset,
        "maximum_onset_repair_offset": ONSET_REPAIR_MAX_OFFSET,
    }
    report["checks"].update(late_checks)
    report["passed"] = all(report["checks"].values())
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"v334_passed": report["passed"], "late_checks": late_checks,
                      "late_phase_fallback": report["late_phase_fallback"]}, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
