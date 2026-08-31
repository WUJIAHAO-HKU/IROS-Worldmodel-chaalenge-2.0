#!/usr/bin/env python3
"""Extend the frozen v326 contract with phase-aligned successor checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import test_v324_phase_guarded_terminal as inherited
from audit_v310_full_mirror_causal_gate import make_counterfactual
from wam_pipeline.v317_batched_sparse_failure_terminal_runtime import Track2V317BatchedSparseFailureTerminal
from wam_pipeline.v326_blended_phase_terminal_runtime import Track2V326BlendedPhaseTerminal
from wam_pipeline.v332_phase_aligned_successor_runtime import Track2V332PhaseAlignedSuccessor


def temporal_delta_error(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.abs(
        np.diff(prediction.astype(np.float32), axis=0)
        - np.diff(target.astype(np.float32), axis=0)
    ).mean())


def main() -> int:
    inherited.Track2V324PhaseGuardedTerminal = Track2V332PhaseAlignedSuccessor
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
    path = windows / "episode6_00100.npz"
    with np.load(path, allow_pickle=False) as values:
        context = values["context_frames"].astype(np.uint8)
        history = values["history_actions"].astype(np.float32)
        future = values["future_actions"].astype(np.float32)
        target = values["target_frames"].astype(np.uint8)
    value_seed = inherited.seed(path)
    prompt = "adjust bottle"
    candidate_runtime = Track2V332PhaseAlignedSuccessor(
        checkpoint, library, device, action_gate, phase_gate
    )
    v326_runtime = Track2V326BlendedPhaseTerminal(
        checkpoint, library, device, action_gate, phase_gate
    )
    v317_runtime = Track2V317BatchedSparseFailureTerminal(
        checkpoint, library, device, action_gate
    )
    base, _ = candidate_runtime._nearest_clean(context, history, future)
    phase_ready, phase_episode, phase_start, phase_onset = candidate_runtime._phase(base)
    probability = candidate_runtime._probability(history, future)
    signature = candidate_runtime._signature(history, future, probability)
    coherent = candidate_runtime._coherent_phase_override(
        context, history, future, probability, signature
    )
    candidate = candidate_runtime.predict(context, history, future, value_seed, prompt)
    v326 = v326_runtime.predict(context, history, future, value_seed, prompt)
    variants = [
        make_counterfactual(future, history, name)
        for name in ("open_gripper", "static_transport", "reverse_transport")
    ]
    candidate_counterfactuals = candidate_runtime.predict_batch(
        np.repeat(context[None], 3, axis=0),
        np.repeat(history[None], 3, axis=0),
        np.stack(variants),
        np.repeat(value_seed, 3),
        [prompt] * 3,
    )
    v317_counterfactuals = v317_runtime.predict_batch(
        np.repeat(context[None], 3, axis=0),
        np.repeat(history[None], 3, axis=0),
        np.stack(variants),
        np.repeat(value_seed, 3),
        [prompt] * 3,
    )
    candidate_rgb = float(np.abs(candidate[-5:].astype(np.int16) - target[-5:].astype(np.int16)).mean())
    v326_rgb = float(np.abs(v326[-5:].astype(np.int16) - target[-5:].astype(np.int16)).mean())
    candidate_temporal = temporal_delta_error(candidate[-5:], target[-5:])
    v326_temporal = temporal_delta_error(v326[-5:], target[-5:])
    added = {
        "aligned_transition_override_true": coherent,
        "aligned_transition_phase_ready_and_eligible": phase_ready and candidate_runtime.eligible_episode[phase_episode],
        "aligned_transition_target_is_base_row": bool(
            int(candidate_runtime.row_episode[base]) == phase_episode
            and int(candidate_runtime.row_start[base]) == phase_start
            and phase_start >= phase_onset
        ),
        "aligned_transition_terminal_changes_v326": bool(np.any(candidate[-1] != v326[-1])),
        "aligned_transition_preterminal_bit_exact_v326": bool(np.array_equal(candidate[:-1], v326[:-1])),
        "aligned_next_context_rgb_strictly_improves_v326": candidate_rgb < v326_rgb,
        "aligned_temporal_delta_strictly_improves_v326": candidate_temporal < v326_temporal,
        "counterfactual_full_sequences_bit_exact_v317": bool(
            np.array_equal(candidate_counterfactuals, v317_counterfactuals)
        ),
    }
    report["format"] = "strict-track2-v332-phase-aligned-successor-contract-v1"
    report["candidate"] = "track2-v332-phase-aligned-successor-v326-v317"
    report["inherited_contract"] = "strict-track2-v324-phase-guarded-terminal-contract-v1"
    report["aligned_transition"] = {
        "window": path.name,
        "phase_episode": phase_episode,
        "phase_start": phase_start,
        "phase_onset": phase_onset,
        "candidate_next_context_rgb_mae": candidate_rgb,
        "v326_next_context_rgb_mae": v326_rgb,
        "candidate_temporal_delta_error": candidate_temporal,
        "v326_temporal_delta_error": v326_temporal,
    }
    report["checks"].update(added)
    report["passed"] = all(report["checks"].values())
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"v332_passed": report["passed"], "added_checks": added,
                      "aligned_transition": report["aligned_transition"]}, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
